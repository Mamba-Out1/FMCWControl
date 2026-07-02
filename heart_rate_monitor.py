# ===========================================================================
# 瀹炴椂蹇冪巼鐩戞祴绯荤粺 - 鍚庣鏈嶅姟
# ===========================================================================

import numpy as np
from collections import deque
import math
import serial
import os
try:
    from scipy.signal import butter, filtfilt, find_peaks, medfilt
    _has_scipy = True
except Exception:
    butter = None
    filtfilt = None
    find_peaks = None
    medfilt = None
    _has_scipy = False
try:
    import onnxruntime as ort
    _has_onnxruntime = True
except Exception:
    ort = None
    _has_onnxruntime = False
try:
    import torch
    import torch.nn as nn
    _has_torch = True
except Exception:
    torch = None
    nn = None
    _has_torch = False
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import threading
import time
import traceback


if _has_torch:
    class TorchRadarTransformer(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=128, nhead=4, num_layers=2):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.reg_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.feature_proj = nn.Linear(hidden_dim, 256)

        def forward(self, x):
            x_proj = self.input_proj(x)
            seq_len = x_proj.size(1)
            position = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
            x_proj = x_proj + (position / max(seq_len, 1))
            encoded = self.transformer(x_proj)
            features = encoded.mean(dim=1)
            return self.reg_head(features)
else:
    TorchRadarTransformer = None


class DistilledHeartRateExtractor:
    """Documentation."""
    def __init__(self, fs, heart_band, window_len=128, hop_len=32, model_path=None):
        self.fs = fs
        self.heart_band = heart_band
        self.window_len = window_len
        self.hop_len = hop_len
        self.model_path = model_path
        self.model_loaded = False
        self.model_type = None
        self.model = None
        self.last_model_hr = 0.0
        self.last_filter_mode = 'fallback'
        self.load_model()

    def load_model(self):
        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        candidates.extend(['heart_rate_student.onnx', 'heart_rate_student.pt', 'heart_rate_student.npz'])

        seen = set()
        for path in candidates:
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)

            if path.endswith('.onnx') and _has_onnxruntime:
                try:
                    self.model = ort.InferenceSession(path)
                    self.model_type = 'onnx'
                    self.model_loaded = True
                    self.model_path = path
                    print(f"Loaded KD ONNX model: {path}")
                    return
                except Exception as e:
                    print(f"Failed to load KD ONNX model {path}: {e}")

            if path.endswith('.pt') and _has_torch and TorchRadarTransformer is not None:
                try:
                    model = TorchRadarTransformer()
                    state_dict = torch.load(path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.model = model
                    self.model_type = 'torch'
                    self.model_loaded = True
                    self.model_path = path
                    print(f"Loaded KD PyTorch model: {path}")
                    return
                except Exception as e:
                    print(f"Failed to load KD PyTorch model {path}: {e}")

            if path.endswith('.npz'):
                try:
                    self.model = np.load(path, allow_pickle=True)
                    self.model_type = 'npz'
                    self.model_loaded = True
                    self.model_path = path
                    print(f"Loaded KD NumPy weights: {path}")
                    return
                except Exception as e:
                    print(f"Failed to load KD NumPy weights {path}: {e}")

        print("KD model not loaded; using adaptive fallback filter.")
        self.model_loaded = False

    def make_windows(self, signal):
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            signal = signal.flatten()
        n = len(signal)
        if n < self.window_len:
            return np.array([signal])
        step = self.hop_len
        windows = []
        for start in range(0, max(1, n - self.window_len + 1), step):
            windows.append(signal[start:start + self.window_len])
        if windows and len(windows[-1]) < self.window_len:
            last = signal[-self.window_len:]
            windows[-1] = last
        return np.vstack(windows)

    def overlap_add(self, windows, output_len):
        if windows.size == 0:
            return np.zeros(output_len, dtype=float)
        if windows.ndim == 1:
            return windows[:output_len]
        n_windows, win_len = windows.shape
        out = np.zeros(output_len, dtype=float)
        weight = np.zeros(output_len, dtype=float)
        for i in range(n_windows):
            start = i * self.hop_len
            end = start + win_len
            if end > output_len:
                end = output_len
                segment = windows[i, :end - start]
            else:
                segment = windows[i]
            out[start:end] += segment
            weight[start:end] += 1.0
        weight[weight == 0] = 1.0
        return out / weight

    def predict_hr_from_features(self, radar_features):
        features = np.asarray(radar_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 64 or features.shape[0] < 2:
            return 0.0

        if self.model_loaded and self.model_type == 'onnx':
            try:
                input_name = self.model.get_inputs()[0].name
                result = self.model.run(None, {input_name: features[np.newaxis, :, :]})
                hr = float(np.asarray(result[0]).reshape(-1)[0])
                if 40.0 <= hr <= 180.0:
                    self.last_model_hr = hr
                    return hr
            except Exception as e:
                print(f'KD ONNX HR inference failed: {e}')

        if self.model_loaded and self.model_type == 'torch' and _has_torch:
            try:
                with torch.no_grad():
                    x = torch.from_numpy(features[np.newaxis, :, :]).float()
                    result = self.model(x)
                hr = float(result.detach().cpu().numpy().reshape(-1)[0])
                if 40.0 <= hr <= 180.0:
                    self.last_model_hr = hr
                    return hr
            except Exception as e:
                print(f'KD PyTorch HR inference failed: {e}')

        return 0.0

    def simple_bandpass_1d(self, signal, lowcut, highcut):
        signal = np.asarray(signal, dtype=float)
        if signal.size < 4:
            return signal
        fft_data = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / self.fs)
        mask = (freqs >= lowcut) & (freqs <= highcut)
        fft_data[~mask] = 0
        return np.fft.irfft(fft_data, n=signal.size)

    def adaptive_filter(self, signal, model_hr=0.0):
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            return signal

        signal = signal - np.median(signal)
        base_filtered = self._bandpass(signal, self.heart_band[0], self.heart_band[1])

        if model_hr and 40.0 <= model_hr <= 180.0:
            center = model_hr / 60.0
            low = max(self.heart_band[0], center - 0.38)
            high = min(self.heart_band[1], center + 0.38)
            if high <= low + 0.05:
                low, high = self.heart_band
        else:
            low, high = self.heart_band

        kd_filtered = self._bandpass(signal, low, high)
        base_std = float(np.std(base_filtered))
        kd_std = float(np.std(kd_filtered))
        tail_len = min(len(signal), max(32, int(self.fs * 5)))
        base_tail_std = float(np.std(base_filtered[-tail_len:]))
        kd_tail_std = float(np.std(kd_filtered[-tail_len:]))

        if base_std < 1e-6:
            filtered = kd_filtered
            self.last_filter_mode = 'kd'
        elif kd_std < 0.15 * base_std or (base_tail_std > 1e-6 and kd_tail_std < 0.10 * base_tail_std):
            filtered = base_filtered
            self.last_filter_mode = 'base_guard'
        else:
            gain = base_std / max(kd_std, 1e-6)
            kd_filtered = kd_filtered * np.clip(gain, 0.5, 3.0)
            filtered = 0.55 * kd_filtered + 0.45 * base_filtered
            self.last_filter_mode = 'kd_blend'

        return self.post_process(filtered)

    def _bandpass(self, signal, low, high):
        if _has_scipy and butter is not None and filtfilt is not None and signal.size > int(self.fs * 3):
            try:
                b, a = butter(3, [low / (0.5 * self.fs), high / (0.5 * self.fs)], btype='band')
                return filtfilt(b, a, signal)
            except Exception:
                return self.simple_bandpass_1d(signal, low, high)
        return self.simple_bandpass_1d(signal, low, high)

    def predict(self, signal, radar_features=None):
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            return np.array([], dtype=float)
        model_hr = self.predict_hr_from_features(radar_features) if radar_features is not None else 0.0
        if model_hr > 0:
            return self.adaptive_filter(signal, model_hr)
        windows = self.make_windows(signal)
        if self.model_loaded and self.model_type == 'onnx':
            preds = []
            input_name = self.model.get_inputs()[0].name
            for w in windows:
                x = w.astype(np.float32).reshape(1, 1, -1)
                try:
                    result = self.model.run(None, {input_name: x})
                    pred = np.squeeze(result[0])
                except Exception as e:
                    print(f'KD ONNX 鎺ㄧ悊澶辫触: {e}')
                    pred = w
                preds.append(pred)
            preds = np.vstack(preds)
        elif self.model_loaded and self.model_type == 'npz':
            preds = self.simple_npz_predict(windows)
        else:
            preds = self.simple_fallback(windows)

        output = self.overlap_add(preds, signal.size)
        return self.post_process(output)

    def simple_npz_predict(self, windows):
        if not hasattr(self.model, 'files') or 'W0' not in self.model:
            return self.simple_fallback(windows)
        output = []
        for w in windows:
            x = (w - np.mean(w)) / (np.std(w) + 1e-6)
            for i in range(10):
                key_w = f'W{i}'
                key_b = f'b{i}'
                if key_w in self.model and key_b in self.model:
                    x = np.dot(x, self.model[key_w]) + self.model[key_b]
                    x = np.tanh(x)
                else:
                    break
            output.append(x)
        return np.vstack(output)

    def simple_fallback(self, windows):
        preds = []
        for w in windows:
            w = w - np.mean(w)
            freqs = np.fft.rfftfreq(w.size, d=1.0 / self.fs)
            fft_w = np.fft.rfft(w)
            mask = (freqs >= self.heart_band[0]) & (freqs <= self.heart_band[1])
            fft_w[~mask] = 0
            pred = np.fft.irfft(fft_w, n=w.size)
            preds.append(pred)
        return np.vstack(preds)

    def post_process(self, waveform):
        if waveform.size == 0:
            return waveform
        waveform = waveform - np.median(waveform)
        waveform = np.clip(waveform, -3.0, 3.0)
        if _has_scipy and butter is not None and filtfilt is not None:
            b, a = butter(3, np.array(self.heart_band) / (0.5 * self.fs), btype='band')
            waveform = filtfilt(b, a, waveform)
        if _has_scipy and medfilt is not None:
            try:
                waveform = medfilt(waveform, kernel_size=7)
            except Exception:
                pass
        return waveform

    def estimate_hr(self, waveform):
        if waveform.size < int(self.fs * 2):
            return 0.0
        waveform = waveform - np.mean(waveform)
        freqs = np.fft.rfftfreq(waveform.size, d=1.0 / self.fs)
        spectrum = np.abs(np.fft.rfft(waveform))
        mask = (freqs >= self.heart_band[0]) & (freqs <= self.heart_band[1])
        if not np.any(mask):
            return 0.0
        band_freqs = freqs[mask]
        band_spec = spectrum[mask]
        if band_spec.size == 0:
            return 0.0
        peak = band_freqs[np.argmax(band_spec)]
        return float(peak * 60.0)


app = Flask(__name__)
CORS(app)

# 鍚敤Flask璋冭瘯妯″紡鐨勮缁嗛敊璇?
app.config['PROPAGATE_EXCEPTIONS'] = True

# 鍏ㄥ眬鏁版嵁瀛樺偍
class RadarDataProcessor:
    def __init__(self):
        # 淇″彿澶勭悊鍙傛暟
        self.fs = 20  # 閲囨牱鐜?(Hz) - 涓巉rame_repetition_time瀵瑰簲
        self.heart_rate_band = [0.8, 2.5]  # 蹇冪巼棰戝甫 (48-150 BPM)
        self.respiratory_band = [0.15, 0.5]  # 鍛煎惛鐜囬甯?(9-30 BPM)
        
        self.heart_rate_buffer = deque(maxlen=500)  # 瀛樺偍500涓績鐜囨暟鎹偣
        self.respiratory_buffer = deque(maxlen=500)  # 瀛樺偍500涓懠鍚哥巼鏁版嵁鐐?
        self.raw_phase_buffer = deque(maxlen=1000)  # 鍘熷鐩镐綅鏁版嵁缂撳啿
        self.radar_feature_buffer = deque(maxlen=160)
        self.raw_data_buffer = deque(maxlen=200)  # 鍘熷闆疯揪鏁版嵁缂撳啿锛堢敤浜庢樉绀猴級
        self.latest_range_doppler = None  # 鏈€鏂拌寖鍥?澶氭櫘鍕掑浘骞呭害
        self.range_axis = []
        self.doppler_axis = []
        self.target_min_m = 0.20
        self.target_max_m = 1.20
        self.last_target_bin = None
        self.last_target_distance_m = 0.0
        self.last_target_snr = 0.0
        self.target_present = False
        self.target_lost_frames = 0
        # BMD101/ECG 涓插彛鐩稿叧
        self.ecg_buffer = deque(maxlen=2000)  # 瀛樺偍瑙ｆ瀽鍚庣殑 ECG 鏍锋湰
        self.ecg_time_buffer = deque(maxlen=2000)  # 瀵瑰簲鏃堕棿鎴?
        self.serial_port = "COM5"
        self.serial_baud = 57600
        self.serial_thread = None
        self.serial_running = False
        self.serial_lock = threading.Lock()
        self.current_heart_rate = 0  # 褰撳墠蹇冪巼BPM
        self.current_respiratory_rate = 0  # 褰撳墠鍛煎惛鐜嘊PM
        self.latest_kd_waveform = []
        self.kd_wave_buffer = deque(maxlen=500)
        self.current_kd_hr = 0.0
        self.kd_status = 'fallback'
        self.kd_processor = DistilledHeartRateExtractor(fs=self.fs, heart_band=self.heart_rate_band)
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0  # 甯ц鏁板櫒
        self.error_message = ""  # 閿欒淇℃伅
        self.last_frame_time = 0  # 涓婁竴甯ф椂闂?
        
        # 闆疯揪閰嶇疆
        self.config = FmcwSimpleSequenceConfig(
            frame_repetition_time_s=0.05,  # 20Hz甯х巼
            chirp_repetition_time_s=500e-6,
            num_chirps=16,
            tdm_mimo=False,
            chirp=FmcwSequenceChirp(
                start_frequency_Hz=59e9,
                end_frequency_Hz=61e9,
                sample_rate_Hz=2e6,
                num_samples=128,
                rx_mask=7,
                tx_mask=1,
                tx_power_level=31,
                lp_cutoff_Hz=500000,
                hp_cutoff_Hz=80000,
                if_gain_dB=30,
            ),
        )
        
    def extract_phase_from_frame(self, frame_data):
        """Extract phase and display value from one radar frame."""
        try:
            frame = np.asarray(frame_data)
            if frame.ndim != 3:
                raise ValueError(f"unexpected frame shape: {frame.shape}")

            rx_count, chirp_count, sample_count = frame.shape
            centered = frame - np.mean(frame, axis=2, keepdims=True)
            window = np.hanning(sample_count).reshape(1, 1, -1)
            range_fft = np.fft.fft(centered * window, axis=2)[..., :sample_count // 2]
            magnitude = np.mean(np.abs(range_fft), axis=(0, 1))

            range_axis = np.asarray(self.get_range_axis(sample_count), dtype=float)
            if range_axis.size != magnitude.size:
                range_axis = range_axis[:magnitude.size]
            roi_mask = (range_axis >= self.target_min_m) & (range_axis <= self.target_max_m)
            roi_indices = np.where(roi_mask)[0]
            if roi_indices.size == 0:
                roi_indices = np.arange(2, magnitude.size)

            roi_magnitude = magnitude[roi_indices]
            noise_floor = float(np.median(roi_magnitude) + 1e-6)
            peak_local = int(np.argmax(roi_magnitude))
            peak_bin = int(roi_indices[peak_local])
            peak_value = float(roi_magnitude[peak_local])

            if self.last_target_bin is not None:
                track_min = max(int(roi_indices[0]), self.last_target_bin - 2)
                track_max = min(int(roi_indices[-1]), self.last_target_bin + 2)
                track_indices = np.arange(track_min, track_max + 1)
                if track_indices.size > 0:
                    track_values = magnitude[track_indices]
                    track_peak_local = int(np.argmax(track_values))
                    track_bin = int(track_indices[track_peak_local])
                    track_value = float(track_values[track_peak_local])
                    if track_value > 0.70 * peak_value:
                        peak_bin = track_bin
                        peak_value = track_value

            snr = peak_value / noise_floor
            self.last_target_bin = peak_bin
            self.last_target_distance_m = float(range_axis[peak_bin]) if peak_bin < range_axis.size else 0.0
            self.last_target_snr = float(snr)
            self.target_present = bool(snr >= 1.35)

            target_complex = range_fft[:, :, peak_bin]
            phase = float(np.angle(np.mean(target_complex)))
            raw_value = float(20.0 * np.log10(peak_value + 1e-6))
            return phase, raw_value
        except Exception as e:
            print(f"鎻愬彇鐩镐綅閿欒: {e}")
            self.target_present = False
            return 0.0, 0.0
    
    def extract_kd_radar_feature(self, frame_data):
        try:
            frame_fft = np.abs(np.fft.fft(frame_data, axis=-1))
            n_freqs = min(64, frame_fft.shape[-1] // 2)
            energy = np.mean(frame_fft[:, :, :n_freqs], axis=(0, 1))
            if len(energy) < 64:
                energy = np.interp(np.linspace(0, len(energy) - 1, 64), np.arange(len(energy)), energy)
            else:
                energy = energy[:64]
            energy = np.log1p(energy.astype(np.float32))
            std = np.std(energy)
            if std > 1e-6:
                energy = (energy - np.mean(energy)) / std
            return energy.astype(np.float32)
        except Exception as e:
            print(f"KD radar feature extraction error: {e}")
            return np.zeros(64, dtype=np.float32)

    def compute_range_doppler_cube(self, frame_data, chirp_rt, half_range=True):
        """Documentation."""
        rx, num_chirps, num_samples = frame_data.shape
        if num_chirps < 2 or num_samples < 2:
            return np.zeros((rx, num_chirps, num_samples // 2), dtype=float)

        window_range = np.hanning(num_samples)
        window_doppler = np.hanning(num_chirps)

        rng = np.fft.fft(frame_data * window_range[np.newaxis, np.newaxis, :], axis=2)
        if half_range:
            rng = rng[..., : num_samples // 2]

        dop = np.fft.fftshift(
            np.fft.fft(rng * window_doppler[np.newaxis, :, np.newaxis], axis=1),
            axes=1,
        )
        return np.abs(dop)

    def get_range_axis(self, num_samples):
        """Documentation."""
        start_f = self.config.chirp.start_frequency_Hz
        end_f = self.config.chirp.end_frequency_Hz
        bandwidth = max(end_f - start_f, 1.0)
        num_range = num_samples // 2
        range_resolution = 3e8 / (2.0 * bandwidth)
        return list(np.arange(num_range) * range_resolution)

    def get_doppler_axis(self, num_chirps):
        """Documentation."""
        return list(np.fft.fftshift(np.fft.fftfreq(num_chirps, d=self.config.chirp_repetition_time_s)))

    def unwrap_phase(self, phase_buffer):
        """Documentation."""
        phases = np.array(phase_buffer)
        unwrapped = np.unwrap(phases)
        return unwrapped

    def clear_vital_buffers(self):
        self.raw_phase_buffer.clear()
        self.heart_rate_buffer.clear()
        self.respiratory_buffer.clear()
        self.radar_feature_buffer.clear()
        self.kd_wave_buffer.clear()
        self.latest_kd_waveform = []
        self.current_heart_rate = 0.0
        self.current_respiratory_rate = 0.0
        self.current_kd_hr = 0.0

    def _serial_reader_loop(self):
        """Documentation."""
        # 灏濊瘯鎵撳紑涓插彛锛屽娆￠噸璇曚互搴斿涓存椂鏉冮檺鎴栧崰鐢ㄩ棶棰?
        max_open_attempts = 6
        attempt = 0
        ser = None
        while attempt < max_open_attempts:
            try:
                ser = serial.Serial(self.serial_port, self.serial_baud, timeout=1)
                break
            except PermissionError as pe:
                attempt += 1
                print(f"灏濊瘯鎵撳紑涓插彛 {self.serial_port} 琚嫆缁?(PermissionError)锛屽皾璇?{attempt}/{max_open_attempts}: {pe}")
                # 缁欑敤鎴锋彁绀猴細鍙兘绔彛琚叾浠栫▼搴忓崰鐢ㄦ垨鏉冮檺涓嶈冻
                if attempt == max_open_attempts:
                    print("涓ラ噸: 鏃犳硶鎵撳紑涓插彛锛屽缓璁細\n 1) 纭娌℃湁鍏朵粬绋嬪簭鍗犵敤 COM 绔彛锛堜緥濡備覆鍙ｇ粓绔€両DE 缁堢锛夈€俓n 2) 鍦ㄥ懡浠よ涓繍琛屼笌鍙敤鐨勭畝鍗曡剼鏈獙璇佺鍙ｅ彲鐢ㄣ€俓n 3) 浠ョ鐞嗗憳韬唤杩愯姝ゆ湇鍔″皾璇曘€俓n")
                    return
                time.sleep(0.7)
                continue
            except Exception as e:
                attempt += 1
                print(f"灏濊瘯鎵撳紑涓插彛 {self.serial_port} 澶辫触锛屽皾璇?{attempt}/{max_open_attempts}: {e}")
                if attempt == max_open_attempts:
                    return
                time.sleep(0.7)

        if ser is None:
            print(f"鏃犳硶鎵撳紑涓插彛 {self.serial_port}锛屽凡鏀惧純")
            return

        print(f"涓插彛宸叉墦寮€: {self.serial_port} (PID={os.getpid()})")
        self.serial_running = True
        try:
            while self.serial_running:
                try:
                    line = ser.readline()
                    if not line:
                        continue
                    # 鍘婚櫎甯歌鐨勫抚濉厖/鍚屾瀛楄妭 0xAA 0xAA锛屽啀鎸夊皬绔?(lo,hi) 瑙ｆ瀽 16-bit 鏈夌鍙锋牱鏈?
                    b = line.replace(b'\xaa\xaa', b'')
                    vals = []
                    for i in range(0, len(b) - 1, 2):
                        lo = b[i]
                        hi = b[i + 1]
                        v = lo | (hi << 8)
                        
                        if v >= 0x8000:
                            v -= 0x10000
                        vals.append(int(v))

                    if vals:
                        now = time.time()
                        with self.serial_lock:
                            for vv in vals:
                                # 鍙姞鍏ラ潪闆舵垨鏄庢樉鍊硷紝閬垮厤绾帶鍒跺寘閫犳垚澶ч噺闆?
                                self.ecg_buffer.append(vv)
                                self.ecg_time_buffer.append(now)
                except Exception as inner:
                    print(f"涓插彛璇诲彇/瑙ｆ瀽閿欒: {inner}")
                    time.sleep(0.01)
        finally:
            try:
                ser.close()
            except Exception:
                pass
            self.serial_running = False

    def start_serial_reader(self):
        if self.serial_thread and self.serial_thread.is_alive():
            return
        self.serial_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        self.serial_thread.start()

    def stop_serial_reader(self):
        self.serial_running = False
        if self.serial_thread:
            self.serial_thread.join(timeout=0.5)
    
    def butter_bandpass_coeffs(self, lowcut, highcut, fs, order=2):
        """Documentation."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # 绠€鍖栫殑浜岄樁宸寸壒娌冩柉婊ゆ尝鍣?
        # 浣跨敤鍙岀嚎鎬у彉鎹㈣繎浼?
        w1 = 2 * math.pi * lowcut
        w2 = 2 * math.pi * highcut
        
        return (low, high)
    
    def simple_bandpass_filter(self, data, lowcut, highcut, fs):
        """Documentation."""
        if len(data) < 10:
            return data
            
        # FFT
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/fs)
        
        # 鍒涘缓甯﹂€氭护娉㈠櫒鎺╃爜
        mask = np.zeros(len(freqs), dtype=bool)
        mask |= (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
        
        # 搴旂敤婊ゆ尝鍣?
        fft_filtered = fft_data * mask
        
        # 閫咶FT
        filtered_data = np.fft.ifft(fft_filtered).real
        
        return filtered_data
    
    def moving_average_filter(self, data, window_size=5):
        """Documentation."""
        if len(data) < window_size:
            return data
        
        data_array = np.array(data)
        cumsum = np.cumsum(np.insert(data_array, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    def estimate_rate(self, filtered_data, fs, freq_band):
        """閫氳繃FFT浼拌蹇冪巼鎴栧懠鍚哥巼"""
        if len(filtered_data) < 100:  # 闇€瑕佽冻澶熺殑鏁版嵁鐐?
            return 0
        
        # FFT鍒嗘瀽
        fft_data = np.fft.fft(filtered_data)
        freqs = np.fft.fftfreq(len(filtered_data), 1/fs)
        
        # 鍙彇姝ｉ鐜?
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(fft_data)//2])
        
        # 鍦ㄦ寚瀹氶甯﹀唴鎵惧嘲鍊?
        mask = (positive_freqs >= freq_band[0]) & (positive_freqs <= freq_band[1])
        if not np.any(mask):
            return 0
        
        band_freqs = positive_freqs[mask]
        band_fft = positive_fft[mask]
        
        if len(band_fft) == 0:
            return 0
        
        peak_idx = np.argmax(band_fft)
        peak_freq = band_freqs[peak_idx]
        
        # 杞崲涓烘瘡鍒嗛挓娆℃暟
        rate_bpm = peak_freq * 60
        
        return rate_bpm
    
    def process_radar_data(self):
        """闆疯揪鏁版嵁澶勭悊绾跨▼"""
        try:
            with DeviceFmcw() as device:
                print("闆疯揪SDK鐗堟湰: " + get_version())
                print("璁惧UUID: " + device.get_board_uuid())
                print("浼犳劅鍣ㄧ被鍨? " + str(device.get_sensor_type()))
                
                # 閰嶇疆璁惧
                sequence = device.create_simple_sequence(self.config)
                device.set_acquisition_sequence(sequence)
                
                print("寮€濮嬮噰闆嗘暟鎹?..")
                self.is_running = True
                
                while self.is_running:
                    try:
                        # 鑾峰彇涓€甯ф暟鎹?
                        frame_contents = device.get_next_frame()
                        
                        for frame in frame_contents:
                            # 鎻愬彇鐩镐綅淇℃伅鍜屽師濮嬫暟鎹?
                            phase, raw_value = self.extract_phase_from_frame(frame)
                            kd_feature = self.extract_kd_radar_feature(frame)

                            # 鍚屾椂璁＄畻瀹炴椂鑼冨洿-澶氭櫘鍕掑浘
                            try:
                                rd_cube = self.compute_range_doppler_cube(
                                    frame,
                                    self.config.chirp_repetition_time_s,
                                    half_range=True,
                                )
                                rd0 = rd_cube[0] if rd_cube.shape[0] > 0 else rd_cube
                                rd_db = 20.0 * np.log10(rd0 + 1e-6)
                                self.range_axis = self.get_range_axis(frame.shape[2])
                                self.doppler_axis = self.get_doppler_axis(frame.shape[1])
                                with self.lock:
                                    self.latest_range_doppler = rd_db.tolist()
                            except Exception as rd_error:
                                print(f"Range-Doppler 璁＄畻閿欒: {rd_error}")

                            with self.lock:
                                self.frame_count += 1
                                self.last_frame_time = time.time()
                                self.raw_data_buffer.append(raw_value)

                                if not self.target_present:
                                    self.target_lost_frames += 1
                                    if self.target_lost_frames >= 20:
                                        self.clear_vital_buffers()
                                        self.kd_status = 'no_target'
                                    continue

                                self.target_lost_frames = 0
                                self.raw_phase_buffer.append(phase)
                                self.radar_feature_buffer.append(kd_feature)
                                
                                # 褰撴湁瓒冲鏁版嵁鏃惰繘琛屽鐞?
                                if len(self.raw_phase_buffer) >= 100:
                                    try:
                                        # 鐩镐綅瑙ｇ紶缁?
                                        unwrapped_phase = self.unwrap_phase(self.raw_phase_buffer)
                                        
                                        # 鍘婚櫎鐩存祦鍒嗛噺
                                        unwrapped_phase = unwrapped_phase - np.mean(unwrapped_phase)
                                        
                                        # 蹇冪巼鎻愬彇锛堜娇鐢‵FT甯﹂€氭护娉級
                                        heart_signal = self.simple_bandpass_filter(
                                            unwrapped_phase, 
                                            self.heart_rate_band[0], 
                                            self.heart_rate_band[1], 
                                            self.fs
                                        )
                                        if len(heart_signal) > 0:
                                            self.heart_rate_buffer.append(heart_signal[-1])
                                        
                                        # 鍛煎惛鐜囨彁鍙栵紙浣跨敤FFT甯﹂€氭护娉級
                                        respiratory_signal = self.simple_bandpass_filter(
                                            unwrapped_phase,
                                            self.respiratory_band[0],
                                            self.respiratory_band[1],
                                            self.fs
                                        )
                                        if len(respiratory_signal) > 0:
                                            self.respiratory_buffer.append(respiratory_signal[-1])

                                        # Cross-modal KD 妯″瀷杈撳嚭
                                        try:
                                            feature_seq = np.array(list(self.radar_feature_buffer)[-32:], dtype=np.float32)
                                            kd_wave = self.kd_processor.predict(unwrapped_phase, feature_seq if len(feature_seq) >= 2 else None)
                                            if len(kd_wave) > 0:
                                                edge_guard = min(max(3, int(self.fs * 0.5)), max(1, len(kd_wave) // 4))
                                                stable_sample = kd_wave[-edge_guard]
                                                self.kd_wave_buffer.append(float(stable_sample))
                                                self.latest_kd_waveform = [float(x) for x in list(self.kd_wave_buffer)[-500:]]
                                            kd_hr_wave = np.array(self.latest_kd_waveform, dtype=float)
                                            self.current_kd_hr = self.kd_processor.last_model_hr or self.kd_processor.estimate_hr(kd_hr_wave)
                                            self.kd_status = self.kd_processor.last_filter_mode if self.kd_processor.model_loaded else 'fallback'
                                        except Exception as kd_error:
                                            print(f"KD 妯″瀷瀹炴椂澶勭悊閿欒: {kd_error}")
                                            self.kd_status = 'error'
                                        
                                        # 浼拌褰撳墠蹇冪巼鍜屽懠鍚哥巼
                                        if len(self.heart_rate_buffer) >= 100:
                                            self.current_heart_rate = self.estimate_rate(
                                                list(self.heart_rate_buffer)[-200:],
                                                self.fs,
                                                self.heart_rate_band
                                            )
                                            
                                        if len(self.respiratory_buffer) >= 100:
                                            self.current_respiratory_rate = self.estimate_rate(
                                                list(self.respiratory_buffer)[-200:],
                                                self.fs,
                                                self.respiratory_band
                                            )
                                    except Exception as filter_error:
                                        print(f"婊ゆ尝澶勭悊閿欒: {filter_error}")
                                        self.error_message = str(filter_error)
                        
                        time.sleep(0.001)  # 鐭殏寤惰繜閬垮厤CPU杩囪浇
                    
                    except Exception as frame_error:
                        print(f"甯у鐞嗛敊璇? {frame_error}")
                        self.error_message = str(frame_error)
                        time.sleep(0.1)
                    
        except Exception as e:
            print(f"闆疯揪鏁版嵁澶勭悊閿欒: {e}")
            print(traceback.format_exc())
            self.error_message = str(e)
            self.is_running = False
    
    def start(self):
        """鍚姩鏁版嵁閲囬泦"""
        if not self.is_running:
            thread = threading.Thread(target=self.process_radar_data, daemon=True)
            thread.start()
            # 鍚姩涓插彛璇诲彇绾跨▼
            try:
                self.start_serial_reader()
            except Exception as e:
                print(f"鍚姩涓插彛璇诲彇澶辫触: {e}")
    
    def stop(self):
        """鍋滄鏁版嵁閲囬泦"""
        self.is_running = False
        # 鍋滄涓插彛璇诲彇
        try:
            self.stop_serial_reader()
        except Exception as e:
            print(f"鍋滄涓插彛璇诲彇澶辫触: {e}")
    
    def get_data(self):
        """鑾峰彇褰撳墠鏁版嵁鐢ㄤ簬API鍝嶅簲"""
        with self.lock:
            # 纭繚杩斿洖鐨勬暟鎹兘鏄疨ython鍘熺敓绫诲瀷锛岄伩鍏岼SON搴忓垪鍖栭敊璇?
            heart_waveform = [float(x) for x in list(self.heart_rate_buffer)[-100:]] if self.heart_rate_buffer else []
            resp_waveform = [float(x) for x in list(self.respiratory_buffer)[-100:]] if self.respiratory_buffer else []
            raw_waveform = [float(x) for x in list(self.raw_data_buffer)[-100:]] if self.raw_data_buffer else []
            # 鑾峰彇 ECG 鏁版嵁鐨勫壇鏈紙浣跨敤鍗曠嫭閿佷互鍑忓皯涓庨浄杈惧鐞嗙殑绔炰簤锛?
            with self.serial_lock:
                ecg_raw_list = list(self.ecg_buffer)[-2000:]
                ecg_time_list = list(self.ecg_time_buffer)[-2000:]

            # 涓哄墠绔樉绀虹敓鎴愪竴涓钩婊?涓嬮噰鏍风増鏈紙鍙€夛級
            ecg_display = []
            if len(ecg_raw_list) > 0:
                # 绠€鍗曞幓鐩存祦骞跺仛鐭粦鍔ㄥ钩鍧囦互鍑忓皯鍣偣
                arr = np.array(ecg_raw_list, dtype=float)
                arr = arr - np.mean(arr)
                window = 3
                if arr.size >= window:
                    cumsum = np.cumsum(np.insert(arr, 0, 0))
                    smooth = (cumsum[window:] - cumsum[:-window]) / window
                    # 鍙栨渶鏂?500 鐐圭敤浜庡睍绀?
                    ecg_display = [float(x) for x in smooth[-500:]]
                else:
                    ecg_display = [float(x) for x in arr[-500:]]
            # R 宄版娴嬩笌蹇冪巼浼拌
            ecg_hr = 0.0
            try:
                if len(ecg_raw_list) >= 60 and len(ecg_time_list) >= len(ecg_raw_list):
                    times = np.array(ecg_time_list[-len(ecg_raw_list):])
                    dt = np.median(np.diff(times)) if len(times) > 1 else 0.004
                    fs_est = 1.0 / dt if dt > 0 else 250.0

                    data_arr = np.array(ecg_raw_list[-len(times):], dtype=float)
                    data_arr = data_arr - np.mean(data_arr)

                    # Butterworth 甯﹂€?0.5-40Hz
                    lowcut = 0.5
                    highcut = 40.0
                    if _has_scipy and butter is not None and filtfilt is not None:
                        b, a = butter(4, [lowcut / (0.5 * fs_est), highcut / (0.5 * fs_est)], btype='band')
                        filtered = filtfilt(b, a, data_arr)
                    else:
                        freqs = np.fft.fftfreq(len(data_arr), d=1.0/fs_est)
                        fft_data = np.fft.fft(data_arr)
                        mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
                        fft_data[~mask] = 0
                        filtered = np.fft.ifft(fft_data).real
                        

                    # 骞虫粦鏄剧ず
                    window = max(3, int(fs_est * 0.005))
                    if filtered.size >= window:
                        cumsum = np.cumsum(np.insert(filtered, 0, 0))
                        smoothf = (cumsum[window:] - cumsum[:-window]) / window
                        ecg_display = [float(x) for x in smoothf[-500:]] if smoothf.size > 0 else ecg_display

                    # 宄板€兼娴?
                    peaks_idx = []
                    if _has_scipy and find_peaks is not None:
                        min_dist = int(0.3 * fs_est)
                        height_thr = np.percentile(filtered, 75)
                        peaks_idx, _ = find_peaks(filtered, distance=min_dist, height=height_thr)
                    else:
                        thresh = np.mean(filtered) + np.std(filtered) * 0.5
                        last_idx = -9999
                        min_dist = int(0.3 * fs_est)
                        for i in range(1, len(filtered)-1):
                            if filtered[i] > thresh and filtered[i] > filtered[i-1] and filtered[i] > filtered[i+1]:
                                if i - last_idx > min_dist:
                                    peaks_idx.append(i)
                                    last_idx = i

                    if len(peaks_idx) > 1:
                        peak_times = times[peaks_idx]
                        rr = np.diff(peak_times)
                        if len(rr) > 0:
                            rr_mean = np.mean(rr[-8:])
                            if rr_mean > 0:
                                ecg_hr = 60.0 / rr_mean
            except Exception as e:
                print(f"ECG 瀹炴椂澶勭悊閿欒: {e}")
            
            return {
                'heart_rate_waveform': heart_waveform,
                'respiratory_waveform': resp_waveform,
                'raw_data_waveform': raw_waveform,  # 鏂板锛氬師濮嬫暟鎹尝褰?
                'ecg_waveform': ecg_raw_list[-500:],
                'ecg_display': ecg_display,
                'kd_waveform': self.latest_kd_waveform,
                'kd_hr': float(self.current_kd_hr),
                'kd_status': self.kd_status,
                'target_present': bool(self.target_present),
                'target_distance_m': float(self.last_target_distance_m),
                'target_snr': float(self.last_target_snr),
                'range_doppler_map': self.latest_range_doppler if self.latest_range_doppler is not None else [],
                'range_axis': self.range_axis,
                'doppler_axis': self.doppler_axis,
                'current_heart_rate': float(self.current_heart_rate),
                'current_respiratory_rate': float(self.current_respiratory_rate),
                'frame_count': int(self.frame_count),  # 鏂板锛氬抚璁℃暟
                'is_running': bool(self.is_running),  # 鏂板锛氳繍琛岀姸鎬?
                'error_message': str(self.error_message),  # 鏂板锛氶敊璇俊鎭?
                'buffer_sizes': {  # 鏂板锛氱紦鍐插尯澶у皬
                    'raw_phase': len(self.raw_phase_buffer),
                    'heart_rate': len(self.heart_rate_buffer),
                    'kd': len(self.kd_wave_buffer),
                    'respiratory': len(self.respiratory_buffer),
                    'raw_data': len(self.raw_data_buffer),
                    'ecg': len(self.ecg_buffer)
                },
                'ecg_hr': float(ecg_hr),
                'timestamp': float(time.time())
            }

# 鍒涘缓鍏ㄥ眬澶勭悊鍣ㄥ疄渚?
processor = RadarDataProcessor()

@app.route('/')
def index():
    """Documentation."""
    return render_template('index.html')

@app.route('/mobile')
@app.route('/m')
def mobile_index():
    """Mobile dashboard."""
    return render_template('mobile.html')

@app.route('/api/data')
def get_data():
    """鑾峰彇瀹炴椂鏁版嵁API"""
    try:
        data = processor.get_data()
        return jsonify(data)
    except Exception as e:
        print(f"API閿欒: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/start')
def start_monitoring():
    """鍚姩鐩戞祴"""
    processor.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_monitoring():
    """鍋滄鐩戞祴"""
    processor.stop()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    print("=" * 60)
    print("Radar heart-rate / respiration monitor")
    print("=" * 60)
    print("Starting...")
    processor.start()
    print("Ready: http://localhost:5000")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

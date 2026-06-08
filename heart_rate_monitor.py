# ===========================================================================
# 实时心率监测系统 - 后端服务
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
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import threading
import time
import traceback


class DistilledHeartRateExtractor:
    """跨模态蒸馏模型推理与滤波输出。"""
    def __init__(self, fs, heart_band, window_len=128, hop_len=32, model_path=None):
        self.fs = fs
        self.heart_band = heart_band
        self.window_len = window_len
        self.hop_len = hop_len
        self.model_path = model_path or 'heart_rate_student.onnx'
        self.model_loaded = False
        self.model_type = None
        self.model = None
        self.load_model()

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            if self.model_path.endswith('.onnx') and _has_onnxruntime:
                try:
                    self.model = ort.InferenceSession(self.model_path)
                    self.model_type = 'onnx'
                    self.model_loaded = True
                    print(f'Loaded KD student model from {self.model_path}')
                except Exception as e:
                    print(f'加载 KD ONNX 模型失败: {e}')
                    self.model_loaded = False
            elif self.model_path.endswith('.npz'):
                try:
                    self.model = np.load(self.model_path, allow_pickle=True)
                    self.model_type = 'npz'
                    self.model_loaded = True
                    print(f'Loaded KD student weights from {self.model_path}')
                except Exception as e:
                    print(f'加载 KD 权重失败: {e}')
                    self.model_loaded = False
        else:
            print(f'KD 模型文件未找到: {self.model_path}; 将使用后备滤波器。')
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

    def predict(self, signal):
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            return np.array([], dtype=float)
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
                    print(f'KD ONNX 推理失败: {e}')
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

# 启用Flask调试模式的详细错误
app.config['PROPAGATE_EXCEPTIONS'] = True

# 全局数据存储
class RadarDataProcessor:
    def __init__(self):
        self.heart_rate_buffer = deque(maxlen=500)  # 存储500个心率数据点
        self.respiratory_buffer = deque(maxlen=500)  # 存储500个呼吸率数据点
        self.raw_phase_buffer = deque(maxlen=1000)  # 原始相位数据缓冲
        self.raw_data_buffer = deque(maxlen=200)  # 原始雷达数据缓冲（用于显示）
        self.latest_range_doppler = None  # 最新范围-多普勒图幅度
        self.range_axis = []
        self.doppler_axis = []
        # BMD101/ECG 串口相关
        self.ecg_buffer = deque(maxlen=2000)  # 存储解析后的 ECG 样本
        self.ecg_time_buffer = deque(maxlen=2000)  # 对应时间戳
        self.serial_port = "COM5"
        self.serial_baud = 57600
        self.serial_thread = None
        self.serial_running = False
        self.serial_lock = threading.Lock()
        self.current_heart_rate = 0  # 当前心率BPM
        self.current_respiratory_rate = 0  # 当前呼吸率BPM
        self.latest_kd_waveform = []
        self.current_kd_hr = 0.0
        self.kd_status = 'fallback'
        self.kd_processor = DistilledHeartRateExtractor(fs=self.fs, heart_band=self.heart_rate_band)
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0  # 帧计数器
        self.error_message = ""  # 错误信息
        self.last_frame_time = 0  # 上一帧时间
        
        # 雷达配置
        self.config = FmcwSimpleSequenceConfig(
            frame_repetition_time_s=0.05,  # 20Hz帧率
            chirp_repetition_time_s=500e-6,
            num_chirps=16,
            tdm_mimo=False,
            chirp=FmcwSequenceChirp(
                start_frequency_Hz=59e9,
                end_frequency_Hz=61e9,
                sample_rate_Hz=2e6,
                num_samples=128,
                rx_mask=1,
                tx_mask=1,
                tx_power_level=31,
                lp_cutoff_Hz=500000,
                hp_cutoff_Hz=80000,
                if_gain_dB=30,
            ),
        )
        
        # 信号处理参数
        self.fs = 20  # 采样率 (Hz) - 与frame_repetition_time对应
        self.heart_rate_band = [0.8, 2.5]  # 心率频带 (48-150 BPM)
        self.respiratory_band = [0.15, 0.5]  # 呼吸率频带 (9-30 BPM)
        
    def extract_phase_from_frame(self, frame_data):
        """从雷达帧数据中提取相位信息"""
        try:
            # frame_data 形状: (1, 16, 128) - (天线, 扫频, 采样点)
            # 选择特定距离区间的数据 (假设目标在中间距离)
            range_bin_start = 50  # 距离单元起始
            range_bin_end = 80    # 距离单元结束
            
            # 提取感兴趣区域的数据
            roi_data = frame_data[0, :, range_bin_start:range_bin_end]
            
            # 计算平均幅度最大的距离单元
            amplitudes = np.abs(roi_data)
            mean_amplitudes = np.mean(amplitudes, axis=0)
            best_range_bin = np.argmax(mean_amplitudes)
            
            # 提取该距离单元的数据
            target_data = roi_data[:, best_range_bin]
            
            # 计算相位
            phase = np.angle(target_data.mean())
            
            # 同时保存原始数据的平均值用于显示
            raw_value = float(np.mean(target_data.real))
            
            return phase, raw_value
        except Exception as e:
            print(f"提取相位错误: {e}")
            return 0.0, 0.0
    
    def compute_range_doppler_cube(self, frame_data, chirp_rt, half_range=True):
        """计算范围-多普勒立方体，返回 RX x Doppler x Range 幅度。"""
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
        """计算距离轴（米）。"""
        start_f = self.config.chirp.start_frequency_Hz
        end_f = self.config.chirp.end_frequency_Hz
        bandwidth = max(end_f - start_f, 1.0)
        num_range = num_samples // 2
        range_resolution = 3e8 / (2.0 * bandwidth)
        return list(np.arange(num_range) * range_resolution)

    def get_doppler_axis(self, num_chirps):
        """计算多普勒轴（Hz）。"""
        return list(np.fft.fftshift(np.fft.fftfreq(num_chirps, d=self.config.chirp_repetition_time_s)))

    def unwrap_phase(self, phase_buffer):
        """相位解缠绕"""
        phases = np.array(phase_buffer)
        unwrapped = np.unwrap(phases)
        return unwrapped

    def _serial_reader_loop(self):
        """后台线程：读取串口并解析为 ECG 样本。"""
        # 尝试打开串口，多次重试以应对临时权限或占用问题
        max_open_attempts = 6
        attempt = 0
        ser = None
        while attempt < max_open_attempts:
            try:
                ser = serial.Serial(self.serial_port, self.serial_baud, timeout=1)
                break
            except PermissionError as pe:
                attempt += 1
                print(f"尝试打开串口 {self.serial_port} 被拒绝 (PermissionError)，尝试 {attempt}/{max_open_attempts}: {pe}")
                # 给用户提示：可能端口被其他程序占用或权限不足
                if attempt == max_open_attempts:
                    print("严重: 无法打开串口，建议：\n 1) 确认没有其他程序占用 COM 端口（例如串口终端、IDE 终端）。\n 2) 在命令行中运行与可用的简单脚本验证端口可用。\n 3) 以管理员身份运行此服务尝试。\n")
                    return
                time.sleep(0.7)
                continue
            except Exception as e:
                attempt += 1
                print(f"尝试打开串口 {self.serial_port} 失败，尝试 {attempt}/{max_open_attempts}: {e}")
                if attempt == max_open_attempts:
                    return
                time.sleep(0.7)

        if ser is None:
            print(f"无法打开串口 {self.serial_port}，已放弃")
            return

        print(f"串口已打开: {self.serial_port} (PID={os.getpid()})")
        self.serial_running = True
        try:
            while self.serial_running:
                try:
                    line = ser.readline()
                    if not line:
                        continue
                    # 去除常见的帧填充/同步字节 0xAA 0xAA，再按小端 (lo,hi) 解析 16-bit 有符号样本
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
                                # 只加入非零或明显值，避免纯控制包造成大量零
                                self.ecg_buffer.append(vv)
                                self.ecg_time_buffer.append(now)
                except Exception as inner:
                    print(f"串口读取/解析错误: {inner}")
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
        """计算巴特沃斯带通滤波器系数（简化版）"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # 简化的二阶巴特沃斯滤波器
        # 使用双线性变换近似
        w1 = 2 * math.pi * lowcut
        w2 = 2 * math.pi * highcut
        
        return (low, high)
    
    def simple_bandpass_filter(self, data, lowcut, highcut, fs):
        """简单的带通滤波器（FFT方法）"""
        if len(data) < 10:
            return data
            
        # FFT
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/fs)
        
        # 创建带通滤波器掩码
        mask = np.zeros(len(freqs), dtype=bool)
        mask |= (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
        
        # 应用滤波器
        fft_filtered = fft_data * mask
        
        # 逆FFT
        filtered_data = np.fft.ifft(fft_filtered).real
        
        return filtered_data
    
    def moving_average_filter(self, data, window_size=5):
        """移动平均滤波器"""
        if len(data) < window_size:
            return data
        
        data_array = np.array(data)
        cumsum = np.cumsum(np.insert(data_array, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    def estimate_rate(self, filtered_data, fs, freq_band):
        """通过FFT估计心率或呼吸率"""
        if len(filtered_data) < 100:  # 需要足够的数据点
            return 0
        
        # FFT分析
        fft_data = np.fft.fft(filtered_data)
        freqs = np.fft.fftfreq(len(filtered_data), 1/fs)
        
        # 只取正频率
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(fft_data)//2])
        
        # 在指定频带内找峰值
        mask = (positive_freqs >= freq_band[0]) & (positive_freqs <= freq_band[1])
        if not np.any(mask):
            return 0
        
        band_freqs = positive_freqs[mask]
        band_fft = positive_fft[mask]
        
        if len(band_fft) == 0:
            return 0
        
        peak_idx = np.argmax(band_fft)
        peak_freq = band_freqs[peak_idx]
        
        # 转换为每分钟次数
        rate_bpm = peak_freq * 60
        
        return rate_bpm
    
    def process_radar_data(self):
        """雷达数据处理线程"""
        try:
            with DeviceFmcw() as device:
                print("雷达SDK版本: " + get_version())
                print("设备UUID: " + device.get_board_uuid())
                print("传感器类型: " + str(device.get_sensor_type()))
                
                # 配置设备
                sequence = device.create_simple_sequence(self.config)
                device.set_acquisition_sequence(sequence)
                
                print("开始采集数据...")
                self.is_running = True
                
                while self.is_running:
                    try:
                        # 获取一帧数据
                        frame_contents = device.get_next_frame()
                        
                        for frame in frame_contents:
                            # 提取相位信息和原始数据
                            phase, raw_value = self.extract_phase_from_frame(frame)

                            # 同时计算实时范围-多普勒图
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
                                print(f"Range-Doppler 计算错误: {rd_error}")

                            with self.lock:
                                self.frame_count += 1
                                self.last_frame_time = time.time()
                                self.raw_phase_buffer.append(phase)
                                self.raw_data_buffer.append(raw_value)
                                
                                # 当有足够数据时进行处理
                                if len(self.raw_phase_buffer) >= 100:
                                    try:
                                        # 相位解缠绕
                                        unwrapped_phase = self.unwrap_phase(self.raw_phase_buffer)
                                        
                                        # 去除直流分量
                                        unwrapped_phase = unwrapped_phase - np.mean(unwrapped_phase)
                                        
                                        # 心率提取（使用FFT带通滤波）
                                        heart_signal = self.simple_bandpass_filter(
                                            unwrapped_phase, 
                                            self.heart_rate_band[0], 
                                            self.heart_rate_band[1], 
                                            self.fs
                                        )
                                        if len(heart_signal) > 0:
                                            self.heart_rate_buffer.append(heart_signal[-1])
                                        
                                        # 呼吸率提取（使用FFT带通滤波）
                                        respiratory_signal = self.simple_bandpass_filter(
                                            unwrapped_phase,
                                            self.respiratory_band[0],
                                            self.respiratory_band[1],
                                            self.fs
                                        )
                                        if len(respiratory_signal) > 0:
                                            self.respiratory_buffer.append(respiratory_signal[-1])

                                        # Cross-modal KD 模型输出
                                        try:
                                            kd_wave = self.kd_processor.predict(unwrapped_phase)
                                            self.latest_kd_waveform = [float(x) for x in kd_wave[-500:]]
                                            self.current_kd_hr = self.kd_processor.estimate_hr(kd_wave)
                                            self.kd_status = 'model' if self.kd_processor.model_loaded else 'fallback'
                                        except Exception as kd_error:
                                            print(f"KD 模型实时处理错误: {kd_error}")
                                            self.kd_status = 'error'
                                        
                                        # 估计当前心率和呼吸率
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
                                        print(f"滤波处理错误: {filter_error}")
                                        self.error_message = str(filter_error)
                        
                        time.sleep(0.001)  # 短暂延迟避免CPU过载
                    
                    except Exception as frame_error:
                        print(f"帧处理错误: {frame_error}")
                        self.error_message = str(frame_error)
                        time.sleep(0.1)
                    
        except Exception as e:
            print(f"雷达数据处理错误: {e}")
            print(traceback.format_exc())
            self.error_message = str(e)
            self.is_running = False
    
    def start(self):
        """启动数据采集"""
        if not self.is_running:
            thread = threading.Thread(target=self.process_radar_data, daemon=True)
            thread.start()
            # 启动串口读取线程
            try:
                self.start_serial_reader()
            except Exception as e:
                print(f"启动串口读取失败: {e}")
    
    def stop(self):
        """停止数据采集"""
        self.is_running = False
        # 停止串口读取
        try:
            self.stop_serial_reader()
        except Exception as e:
            print(f"停止串口读取失败: {e}")
    
    def get_data(self):
        """获取当前数据用于API响应"""
        with self.lock:
            # 确保返回的数据都是Python原生类型，避免JSON序列化错误
            heart_waveform = [float(x) for x in list(self.heart_rate_buffer)[-100:]] if self.heart_rate_buffer else []
            resp_waveform = [float(x) for x in list(self.respiratory_buffer)[-100:]] if self.respiratory_buffer else []
            raw_waveform = [float(x) for x in list(self.raw_data_buffer)[-100:]] if self.raw_data_buffer else []
            # 获取 ECG 数据的副本（使用单独锁以减少与雷达处理的竞争）
            with self.serial_lock:
                ecg_raw_list = list(self.ecg_buffer)[-2000:]
                ecg_time_list = list(self.ecg_time_buffer)[-2000:]

            # 为前端显示生成一个平滑/下采样版本（可选）
            ecg_display = []
            if len(ecg_raw_list) > 0:
                # 简单去直流并做短滑动平均以减少噪点
                arr = np.array(ecg_raw_list, dtype=float)
                arr = arr - np.mean(arr)
                window = 3
                if arr.size >= window:
                    cumsum = np.cumsum(np.insert(arr, 0, 0))
                    smooth = (cumsum[window:] - cumsum[:-window]) / window
                    # 取最新 500 点用于展示
                    ecg_display = [float(x) for x in smooth[-500:]]
                else:
                    ecg_display = [float(x) for x in arr[-500:]]
            # R 峰检测与心率估计
            ecg_hr = 0.0
            try:
                if len(ecg_raw_list) >= 60 and len(ecg_time_list) >= len(ecg_raw_list):
                    times = np.array(ecg_time_list[-len(ecg_raw_list):])
                    dt = np.median(np.diff(times)) if len(times) > 1 else 0.004
                    fs_est = 1.0 / dt if dt > 0 else 250.0

                    data_arr = np.array(ecg_raw_list[-len(times):], dtype=float)
                    data_arr = data_arr - np.mean(data_arr)

                    # Butterworth 带通 0.5-40Hz
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

                    # 平滑显示
                    window = max(3, int(fs_est * 0.005))
                    if filtered.size >= window:
                        cumsum = np.cumsum(np.insert(filtered, 0, 0))
                        smoothf = (cumsum[window:] - cumsum[:-window]) / window
                        ecg_display = [float(x) for x in smoothf[-500:]] if smoothf.size > 0 else ecg_display

                    # 峰值检测
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
                print(f"ECG 实时处理错误: {e}")
            
            return {
                'heart_rate_waveform': heart_waveform,
                'respiratory_waveform': resp_waveform,
                'raw_data_waveform': raw_waveform,  # 新增：原始数据波形
                'ecg_waveform': ecg_raw_list[-500:],
                'ecg_display': ecg_display,
                'kd_waveform': self.latest_kd_waveform,
                'kd_hr': float(self.current_kd_hr),
                'kd_status': self.kd_status,
                'range_doppler_map': self.latest_range_doppler if self.latest_range_doppler is not None else [],
                'range_axis': self.range_axis,
                'doppler_axis': self.doppler_axis,
                'current_heart_rate': float(self.current_heart_rate),
                'current_respiratory_rate': float(self.current_respiratory_rate),
                'frame_count': int(self.frame_count),  # 新增：帧计数
                'is_running': bool(self.is_running),  # 新增：运行状态
                'error_message': str(self.error_message),  # 新增：错误信息
                'buffer_sizes': {  # 新增：缓冲区大小
                    'raw_phase': len(self.raw_phase_buffer),
                    'heart_rate': len(self.heart_rate_buffer),
                    'respiratory': len(self.respiratory_buffer),
                    'raw_data': len(self.raw_data_buffer),
                    'ecg': len(self.ecg_buffer)
                },
                'ecg_hr': float(ecg_hr),
                'timestamp': float(time.time())
            }

# 创建全局处理器实例
processor = RadarDataProcessor()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """获取实时数据API"""
    try:
        data = processor.get_data()
        return jsonify(data)
    except Exception as e:
        print(f"API错误: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/start')
def start_monitoring():
    """启动监测"""
    processor.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_monitoring():
    """停止监测"""
    processor.stop()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    print("=" * 60)
    print("雷达心率/呼吸率监测系统")
    print("=" * 60)
    print("启动中...")
    
    # 自动启动数据采集
    processor.start()
    
    print("系统就绪! 请访问: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

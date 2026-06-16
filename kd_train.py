"""
绠€鍖栫増璺ㄦā鎬佽捀棣忚缁冪▼搴?ECG Transformer 鈫?Feature KD 鈫?Radar Transformer 鈫?Heart Rate
鎬绘崯澶? L = L_HR + 0.5*L_feature + 0.2*L_attention
鏀寔鐪熷疄鏁版嵁鍔犺浇: radar.npy (BGT60TR13C) + ECGLog-*.txt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Tuple, Optional, List, Dict
import time
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from scipy.signal import butter, filtfilt, find_peaks


class DataLoader:
    @staticmethod
    def _load_json_if_exists(path: str) -> Dict:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _parse_capture_timestamp(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
        return None

    @staticmethod
    def _find_radar_start_time(radar_npy_path: str) -> Optional[float]:
        radar_dir = os.path.dirname(radar_npy_path)
        parent_dir = os.path.dirname(radar_dir)
        for meta_path in (os.path.join(radar_dir, 'meta.json'), os.path.join(parent_dir, 'meta.json')):
            meta = DataLoader._load_json_if_exists(meta_path)
            start_time = DataLoader._parse_capture_timestamp(meta.get('date_captured'))
            if start_time is not None:
                return start_time
        return None

    @staticmethod
    def _get_radar_frame_period(config: Dict) -> float:
        shape = config.get('device_config', {}).get('fmcw_single_shape', {})
        frame_period = shape.get('frame_repetition_time_s', config.get('frame_repetition_time_s', 0.05))
        frame_period = float(frame_period) if frame_period else 0.05
        return frame_period if frame_period > 0 else 0.05

    @staticmethod
    def build_radar_timestamps(radar_npy_path: str, config: Dict, n_frames: int) -> np.ndarray:
        frame_period = DataLoader._get_radar_frame_period(config)
        start_time = DataLoader._find_radar_start_time(radar_npy_path)
        if start_time is None:
            start_time = 0.0
            print("  Warning: radar start time not found, using relative time from 0s")
        duration = (n_frames - 1) * frame_period if n_frames > 0 else 0.0
        print(f"  Radar time axis: start={start_time:.3f}, frame_period={frame_period:.4f}s, duration={duration:.2f}s")
        return start_time + np.arange(n_frames, dtype=np.float64) * frame_period

    @staticmethod
    def _regularize_ecg_timestamps(timestamps: np.ndarray) -> np.ndarray:
        if len(timestamps) <= 1:
            return timestamps.astype(np.float64)

        timestamps = timestamps.astype(np.float64)
        diffs = np.diff(timestamps)
        if np.all(diffs > 0):
            return timestamps

        starts = [0]
        group_times = [timestamps[0]]
        for idx in range(1, len(timestamps)):
            if timestamps[idx] != group_times[-1]:
                starts.append(idx)
                group_times.append(timestamps[idx])
        starts.append(len(timestamps))

        fixed = np.empty_like(timestamps, dtype=np.float64)
        last_dt = None
        for group_idx in range(len(group_times)):
            start_idx = starts[group_idx]
            end_idx = starts[group_idx + 1]
            count = end_idx - start_idx

            if group_idx + 1 < len(group_times):
                span = group_times[group_idx + 1] - group_times[group_idx]
                dt = span / max(count, 1) if span > 0 else None
            else:
                dt = last_dt

            if dt is None or dt <= 0:
                positive_diffs = diffs[diffs > 0]
                dt = float(np.median(positive_diffs)) if len(positive_diffs) else 1.0 / 250.0
                dt = dt / max(count, 1)

            fixed[start_idx:end_idx] = group_times[group_idx] + np.arange(count) * dt
            last_dt = dt

        return fixed

    """鍔犺浇鍜屽鐞嗙湡瀹炵殑闆疯揪鍜孍CG鏁版嵁"""
    
    @staticmethod
    def load_radar_data(radar_npy_path: str) -> Tuple[np.ndarray, Dict]:
        """
        鍔犺浇BGT60TR13C闆疯揪鏁版嵁 (radar.npy)
        杩斿洖: (radar_frames, config)
        radar_frames shape: (n_frames, n_rx, n_chirps, n_samples)
        """
        if not os.path.exists(radar_npy_path):
            raise FileNotFoundError(f"闆疯揪鏂囦欢涓嶅瓨鍦? {radar_npy_path}")
        
        radar_frames = np.load(radar_npy_path)
        print(f"  鍔犺浇闆疯揪鏁版嵁: {radar_frames.shape}")
        
        # 鍔犺浇閰嶇疆
        config_path = os.path.join(os.path.dirname(radar_npy_path), 'config.json')
        config = DataLoader._load_json_if_exists(config_path)
        
        return radar_frames, config
    
    @staticmethod
    def load_ecg_data(ecg_txt_path: str) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        鍔犺浇ECGLog-*.txt 鏂囦欢
        杩斿洖: (timestamps, adc_values, fs_ecg)
        ECG 鏍煎紡: timestamp ADC HR_4s HR_30s
        """
        if not os.path.exists(ecg_txt_path):
            raise FileNotFoundError(f"ECG鏂囦欢涓嶅瓨鍦? {ecg_txt_path}")
        
        timestamps = []
        adc_values = []
        hr_values = []
        
        with open(ecg_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('timestamp'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        ts_str = parts[0].rstrip(':')
                        ts = float(ts_str)
                        adc = int(parts[1])
                        hr4 = float(parts[2]) if len(parts) >= 3 else np.nan
                        hr30 = float(parts[3]) if len(parts) >= 4 else np.nan
                        hr = hr4 if 30.0 <= hr4 <= 220.0 else hr30
                        timestamps.append(ts)
                        adc_values.append(adc)
                        hr_values.append(hr if 30.0 <= hr <= 220.0 else np.nan)
                    except (ValueError, IndexError):
                        continue
        
        timestamps = DataLoader._regularize_ecg_timestamps(np.array(timestamps, dtype=np.float64))
        adc_values = np.array(adc_values, dtype=np.float32)
        hr_values = np.array(hr_values, dtype=np.float32)
        
        if len(timestamps) > 1:
            positive_diffs = np.diff(timestamps)
            positive_diffs = positive_diffs[positive_diffs > 0]
            dt = np.median(positive_diffs) if len(positive_diffs) else 0
            fs_ecg = 1.0 / dt if dt > 0 else 250.0
        else:
            fs_ecg = 250.0

        print(f"  Loaded ECG data: {len(adc_values)} samples, fs={fs_ecg:.1f} Hz")

        valid_hr_count = int(np.sum(np.isfinite(hr_values)))
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        print(f"  ECG duration={duration:.2f}s, valid HR labels={valid_hr_count}")
        return timestamps, adc_values, fs_ecg, hr_values
    
    @staticmethod
    def extract_radar_features(radar_frames: np.ndarray, config: Dict) -> np.ndarray:
        """Extract 64-D per-frame radar features, shared with realtime inference."""
        features = []
        for frame_idx in range(radar_frames.shape[0]):
            frame = radar_frames[frame_idx]
            frame_fft = np.abs(np.fft.fft(frame, axis=-1))
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
            features.append(energy.astype(np.float32))
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_ecg_features(adc_values: np.ndarray, fs_ecg: float,
                             target_length: int = 100) -> np.ndarray:
        """Legacy ECG feature extraction to a fixed target length."""
        ecg_filtered = DataLoader._filter_ecg(adc_values, fs_ecg)
        window_size = max(8, int(fs_ecg * 0.5))
        hop_size = max(1, window_size // 2)
        features = []
        for start in range(0, max(0, len(ecg_filtered) - window_size), hop_size):
            window = ecg_filtered[start:start + window_size]
            features.append(DataLoader._ecg_window_to_feature(window))

        if not features:
            features = [np.zeros(128, dtype=np.float32)]
        features = np.array(features, dtype=np.float32)

        if len(features) < target_length:
            pad = np.zeros((target_length - len(features), 128), dtype=np.float32)
            features = np.vstack([features, pad])
        else:
            features = features[:target_length]
        return features

    @staticmethod
    def _filter_ecg(adc_values: np.ndarray, fs_ecg: float) -> np.ndarray:
        try:
            b, a = butter(4, [0.5, 40.0], btype='band', fs=fs_ecg)
            ecg_filtered = filtfilt(b, a, adc_values)
        except Exception:
            ecg_filtered = adc_values.astype(np.float32)

        ecg_filtered = ecg_filtered - np.mean(ecg_filtered)
        std = np.std(ecg_filtered)
        if std > 0:
            ecg_filtered = ecg_filtered / std
        return ecg_filtered.astype(np.float32)

    @staticmethod
    def _ecg_window_to_feature(window: np.ndarray) -> np.ndarray:
        feat = np.zeros(128, dtype=np.float32)
        if len(window) == 0:
            return feat

        src_x = np.arange(len(window))
        feat[:32] = np.interp(np.linspace(0, len(window) - 1, 32), src_x, window)

        fft_mag = np.abs(np.fft.rfft(window))
        if len(fft_mag) > 0:
            feat[32:64] = np.interp(np.linspace(0, len(fft_mag) - 1, 32),
                                    np.arange(len(fft_mag)), fft_mag)

        smooth = np.convolve(window, np.ones(5) / 5, mode='same')
        feat[64:96] = np.interp(np.linspace(0, len(smooth) - 1, 32),
                                np.arange(len(smooth)), smooth)

        derivative = np.diff(window, prepend=window[0])
        feat[96:128] = np.interp(np.linspace(0, len(derivative) - 1, 32),
                                 np.arange(len(derivative)), derivative)
        return feat

    @staticmethod
    def extract_ecg_features_at_times(adc_values: np.ndarray, ecg_timestamps: np.ndarray,
                                      fs_ecg: float, target_times: np.ndarray,
                                      window_seconds: float = 0.5) -> np.ndarray:
        ecg_filtered = DataLoader._filter_ecg(adc_values, fs_ecg)
        half_window = window_seconds / 2.0
        features = []

        for target_time in target_times:
            start_time = target_time - half_window
            end_time = target_time + half_window
            left = np.searchsorted(ecg_timestamps, start_time, side='left')
            right = np.searchsorted(ecg_timestamps, end_time, side='right')
            window = ecg_filtered[left:right]

            min_samples = max(8, int(fs_ecg * window_seconds * 0.25))
            if len(window) < min_samples:
                sample_times = target_time + np.linspace(-half_window, half_window, max(8, int(fs_ecg * window_seconds)))
                window = np.interp(sample_times, ecg_timestamps, ecg_filtered)

            features.append(DataLoader._ecg_window_to_feature(window))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _estimate_hr_from_r_peaks(ecg_timestamps: np.ndarray, adc_values: np.ndarray,
                                  fs_ecg: float, target_times: np.ndarray) -> np.ndarray:
        ecg_filtered = DataLoader._filter_ecg(adc_values, fs_ecg)
        min_distance = max(1, int(0.3 * fs_ecg))
        prominence = max(0.25, float(np.std(ecg_filtered)) * 0.5)
        peaks, _ = find_peaks(ecg_filtered, distance=min_distance, prominence=prominence)

        if len(peaks) < 2:
            return np.full(len(target_times), np.nan, dtype=np.float32)

        peak_times = ecg_timestamps[peaks]
        rr = np.diff(peak_times)
        valid = (rr >= 0.3) & (rr <= 2.0)
        if not np.any(valid):
            return np.full(len(target_times), np.nan, dtype=np.float32)

        hr_times = peak_times[:-1][valid] + rr[valid] / 2.0
        hr_values = 60.0 / rr[valid]
        return np.interp(target_times, hr_times, hr_values,
                         left=hr_values[0], right=hr_values[-1]).astype(np.float32)

    @staticmethod
    def generate_hr_labels(target_times: np.ndarray, ecg_timestamps: np.ndarray,
                           adc_values: np.ndarray, fs_ecg: float,
                           hr_values: Optional[np.ndarray] = None) -> np.ndarray:
        labels = np.full(len(target_times), np.nan, dtype=np.float32)

        if hr_values is not None and len(hr_values) == len(ecg_timestamps):
            valid = np.isfinite(hr_values) & (hr_values >= 30.0) & (hr_values <= 220.0)
            if np.any(valid):
                labels = np.interp(target_times, ecg_timestamps[valid], hr_values[valid],
                                   left=hr_values[valid][0], right=hr_values[valid][-1]).astype(np.float32)
                saturation_ratio = float(np.mean(labels >= 219.0))
                if float(np.nanmedian(labels)) > 180.0 or saturation_ratio > 0.2:
                    print("  ECG HR columns look implausible; using R-peak HR labels instead")
                    labels[:] = np.nan

        missing = ~np.isfinite(labels)
        if np.any(missing):
            peak_labels = DataLoader._estimate_hr_from_r_peaks(ecg_timestamps, adc_values, fs_ecg, target_times)
            labels[missing] = peak_labels[missing]

        if np.any(~np.isfinite(labels)):
            fallback = float(np.nanmedian(labels)) if np.any(np.isfinite(labels)) else 75.0
            labels[~np.isfinite(labels)] = fallback

        return np.clip(labels, 30.0, 220.0).astype(np.float32)

    @staticmethod
    def align_and_pair_by_time(radar_features: np.ndarray, radar_timestamps: np.ndarray,
                               ecg_timestamps: np.ndarray, adc_values: np.ndarray,
                               fs_ecg: float, hr_values: Optional[np.ndarray] = None
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        overlap_start = max(float(radar_timestamps[0]), float(ecg_timestamps[0]))
        overlap_end = min(float(radar_timestamps[-1]), float(ecg_timestamps[-1]))
        if overlap_end <= overlap_start:
            raise ValueError(
                f"No radar/ECG time overlap: radar=({radar_timestamps[0]:.3f}, {radar_timestamps[-1]:.3f}), "
                f"ecg=({ecg_timestamps[0]:.3f}, {ecg_timestamps[-1]:.3f})"
            )

        radar_mask = (radar_timestamps >= overlap_start) & (radar_timestamps <= overlap_end)
        aligned_radar = radar_features[radar_mask]
        aligned_times = radar_timestamps[radar_mask]
        if len(aligned_radar) == 0:
            raise ValueError("No radar frames remain after overlap alignment")

        aligned_ecg = DataLoader.extract_ecg_features_at_times(
            adc_values, ecg_timestamps, fs_ecg, aligned_times
        )
        hr_labels = DataLoader.generate_hr_labels(
            aligned_times, ecg_timestamps, adc_values, fs_ecg, hr_values
        )

        print(f"  Time overlap: {overlap_end - overlap_start:.2f}s, paired frames={len(aligned_radar)}")
        return aligned_radar, aligned_ecg, hr_labels
    
    @staticmethod
    def align_and_pair(radar_features: np.ndarray, ecg_features: np.ndarray,
                      radar_timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        瀵归綈闆疯揪鍜孍CG鐗瑰緛鍒扮浉鍚岄暱搴?        杩斿洖: (aligned_radar_features, aligned_ecg_features)
        """
        radar_len = radar_features.shape[0]
        ecg_len = ecg_features.shape[0]
        
        min_len = min(radar_len, ecg_len)
        
        aligned_radar = radar_features[:min_len]
        aligned_ecg = ecg_features[:min_len]
        
        print(f"  瀵归綈鍚? 闆疯揪 {aligned_radar.shape}, ECG {aligned_ecg.shape}")
        
        return aligned_radar, aligned_ecg


class DataSelector:
    """UI file selector."""

    @staticmethod
    def select_data_pairs() -> List[Tuple[str, str]]:
        """Select one or more paired radar.npy and ECGLog txt files."""
        pairs = []
        root = tk.Tk()
        root.withdraw()
        print("\n" + "=" * 60)
        print("Data file selection")
        print("=" * 60)

        pair_idx = 1
        while True:
            print(f"\nPair {pair_idx}")
            print("  Select radar.npy file...")
            radar_path = filedialog.askopenfilename(
                title=f"Select radar.npy for pair {pair_idx}",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            if not radar_path:
                break

            print("  Select matching ECGLog txt file...")
            ecg_path = filedialog.askopenfilename(
                title=f"Select ECG file for pair {pair_idx}",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not ecg_path:
                print("  ECG selection cancelled")
                break

            pairs.append((radar_path, ecg_path))
            print(f"  [OK] Added pair {pair_idx}")
            print(f"    Radar: {os.path.basename(os.path.dirname(radar_path))}/radar.npy")
            print(f"    ECG: {os.path.basename(ecg_path)}")
            pair_idx += 1

            response = input("\nSelect another pair? (y/n): ").strip().lower()
            if response != "y":
                break

        root.destroy()
        if pairs:
            print(f"\n[OK] Selected {len(pairs)} data pair(s)")
        else:
            print("\n[WARN] No data selected")
        return pairs


class ECGTransformer(nn.Module):
    """ECG Transformer鏁欏笀妯″瀷 (绠€鍖栫増)"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, 
                                                   dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.feature_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """杩斿洖棰勬祴鍊煎拰鐗瑰緛"""
        # x shape: (batch, seq_len, input_dim)
        x_proj = self.input_proj(x)
        
        # 鐢熸垚浣嶇疆缂栫爜
        seq_len = x_proj.size(1)
        pos_encoding = self._positional_encoding(seq_len, x_proj.size(-1)).to(x.device)
        x_proj = x_proj + pos_encoding
        
        transformer_output = self.transformer(x_proj)
        features = transformer_output.mean(dim=1)  # 鍏ㄥ眬姹犲寲
        heart_rate = self.reg_head(features)
        
        return heart_rate, features
    
    def _positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """鐢熸垚浣嶇疆缂栫爜 (绠€鍖栫増)"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(1, seq_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_encoding
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """鑾峰彇鎸囧畾灞傜殑娉ㄦ剰鍔涘浘 (鐢ㄤ簬娉ㄦ剰鍔涜捀棣?"""
        x_proj = self.input_proj(x)
        seq_len = x_proj.size(1)
        pos_encoding = self._positional_encoding(seq_len, x_proj.size(-1)).to(x.device)
        x_proj = x_proj + pos_encoding
        
        # 淇濆瓨娉ㄦ剰鍔涘浘
        attention_maps = []
        def hook_fn(module, input, output):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attn_map'):
                attention_maps.append(module.self_attn.attn_map)
        
        # 娉ㄥ唽閽╁瓙
        hooks = []
        for i, layer in enumerate(self.transformer.layers):
            if layer_idx == -1 or i == layer_idx:
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
        
        # 鍓嶅悜浼犳挱
        with torch.no_grad():
            _ = self.transformer(x_proj)
        
        # 绉婚櫎閽╁瓙
        for hook in hooks:
            hook.remove()
        
        return attention_maps[0] if attention_maps else None


class RadarTransformer(nn.Module):
    """Radar Transformer瀛︾敓妯″瀷 (绠€鍖栫増)"""
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                   dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.feature_proj = nn.Linear(hidden_dim, 256)  # 鎶曞奖鍒版暀甯堢壒寰佺淮搴?        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """杩斿洖棰勬祴鍊煎拰鎶曞奖鍚庣殑鐗瑰緛"""
        # x shape: (batch, seq_len, input_dim)
        x_proj = self.input_proj(x)
        
        seq_len = x_proj.size(1)
        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        x_proj = x_proj + (position / seq_len)
        
        transformer_output = self.transformer(x_proj)
        features = transformer_output.mean(dim=1)  # 鍏ㄥ眬姹犲寲
        heart_rate = self.reg_head(features)
        
        # 鎶曞奖鐗瑰緛鐢ㄤ簬钂搁
        projected_features = self.feature_proj(features)
        
        return heart_rate, projected_features
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """鑾峰彇鎸囧畾灞傜殑娉ㄦ剰鍔涘浘 (鐢ㄤ簬娉ㄦ剰鍔涜捀棣?"""
        x_proj = self.input_proj(x)
        seq_len = x_proj.size(1)
        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        x_proj = x_proj + (position / seq_len)
        
        # 淇濆瓨娉ㄦ剰鍔涘浘
        attention_maps = []
        def hook_fn(module, input, output):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attn_map'):
                attention_maps.append(module.self_attn.attn_map)
        
        # 娉ㄥ唽閽╁瓙
        hooks = []
        for i, layer in enumerate(self.transformer.layers):
            if layer_idx == -1 or i == layer_idx:
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
        
        # 鍓嶅悜浼犳挱
        with torch.no_grad():
            _ = self.transformer(x_proj)
        
        # 绉婚櫎閽╁瓙
        for hook in hooks:
            hook.remove()
        
        return attention_maps[0] if attention_maps else None


class KDTrainer:
    """璺ㄦā鎬佽捀棣忚缁冨櫒"""
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        self.teacher = ECGTransformer(input_dim=128, hidden_dim=256, nhead=8, num_layers=4).to(self.device)
        self.student = RadarTransformer(input_dim=64, hidden_dim=128, nhead=4, num_layers=2).to(self.device)
        self.mse_loss = nn.MSELoss()
        self.feature_loss = nn.MSELoss()
        self.attention_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        self.lambda_hr = 1.0
        self.lambda_feature = 0.5
        self.lambda_attention = 0.2

    def train_step(self, radar_data: torch.Tensor, ecg_data: torch.Tensor,
                   heart_rate_labels: torch.Tensor) -> dict:
        radar_data = radar_data.to(self.device)
        ecg_data = ecg_data.to(self.device)
        heart_rate_labels = heart_rate_labels.to(self.device)

        with torch.no_grad():
            _, teacher_features = self.teacher(ecg_data)
            teacher_attention = self.teacher.get_attention_maps(ecg_data)

        student_hr, student_features = self.student(radar_data)
        student_attention = self.student.get_attention_maps(radar_data)

        hr_loss = self.mse_loss(student_hr.squeeze(), heart_rate_labels)
        feature_loss = self.feature_loss(student_features, teacher_features.detach())
        attention_loss = torch.tensor(0.0, device=self.device)
        if teacher_attention is not None and student_attention is not None:
            teacher_attn_mean = teacher_attention.mean(dim=1)
            student_attn_mean = student_attention.mean(dim=1)
            attention_loss = self.attention_loss(student_attn_mean, teacher_attn_mean.detach())

        total_loss = (
            self.lambda_hr * hr_loss +
            self.lambda_feature * feature_loss +
            self.lambda_attention * attention_loss
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'hr_loss': hr_loss.item(),
            'feature_loss': feature_loss.item(),
            'attention_loss': attention_loss.item(),
            'predicted_hr': student_hr.detach().cpu().numpy(),
            'true_hr': heart_rate_labels.cpu().numpy()
        }
    
    def validate(self, radar_data: torch.Tensor, ecg_data: torch.Tensor,
                 heart_rate_labels: torch.Tensor) -> dict:
        """楠岃瘉"""
        self.student.eval()
        with torch.no_grad():
            radar_data = radar_data.to(self.device)
            student_hr, _ = self.student(radar_data)
            
            hr_loss = self.mse_loss(student_hr.squeeze(), heart_rate_labels.to(self.device))
            
            # 璁＄畻蹇冪巼棰勬祴璇樊
            predictions = student_hr.cpu().numpy().flatten()
            labels = heart_rate_labels.cpu().numpy()
            mae = np.mean(np.abs(predictions - labels))
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
            
        self.student.train()
        
        return {
            'val_loss': hr_loss.item(),
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'labels': labels
        }
    
    def save_student_model(self, path: str = 'heart_rate_student.onnx'):
        """Save student model weights and optionally export ONNX."""
        self.student.eval()
        try:
            state_dict = self.student.state_dict()
            torch.save(state_dict, 'heart_rate_student.pt')
            print("Student model weights saved: heart_rate_student.pt")

            npz_path = 'heart_rate_student.npz'
            npz_dict = {}
            for key, value in state_dict.items():
                npz_dict[key] = value.detach().cpu().numpy()
            np.savez(npz_path, **npz_dict)
            print(f"Student model weights saved: {npz_path}")

            try:
                import onnxscript  # noqa: F401
                dummy_input = torch.randn(1, 32, 64).to(self.device)
                torch.onnx.export(
                    self.student,
                    dummy_input,
                    path,
                    input_names=['radar_input'],
                    output_names=['heart_rate'],
                    dynamic_axes={'radar_input': {0: 'batch_size', 1: 'sequence_length'},
                                  'heart_rate': {0: 'batch_size'}},
                    opset_version=12
                )
                print(f"Student model exported: {path}")
            except ModuleNotFoundError:
                print("Skip ONNX export: missing onnxscript. Install with: pip install onnxscript onnx")
            except Exception as exc:
                print(f"Skip ONNX export: {exc}")
        finally:
            self.student.train()
        self.student.train()
    
    def train(self, num_epochs: int = 50, batch_size: int = 32, use_real_data: bool = True):
        """Run KD training on selected real/simulated data."""
        print("Starting cross-modal KD training...")
        print(f"Teacher: ECG Transformer ({sum(p.numel() for p in self.teacher.parameters()):,} params)")
        print(f"Student: Radar Transformer ({sum(p.numel() for p in self.student.parameters()):,} params)")
        print(f"Loss weights: HR={self.lambda_hr}, Feature={self.lambda_feature}, Attention={self.lambda_attention}")

        if use_real_data:
            print("\nLoading real data...")
            data_pairs = DataSelector.select_data_pairs()
            if not data_pairs:
                print("No data selected; using synthetic demo data instead.")
                radar_data, ecg_data, heart_rate_labels = self._generate_synthetic_data(1000, 100)
            else:
                radar_data, ecg_data, heart_rate_labels = self._load_real_data(data_pairs)
        else:
            print("Generating synthetic demo data...")
            radar_data, ecg_data, heart_rate_labels = self._generate_synthetic_data(1000, 100)

        if radar_data is None or len(radar_data) == 0:
            print("Data loading failed")
            return

        radar_data = torch.from_numpy(radar_data).float()
        ecg_data = torch.from_numpy(ecg_data).float()
        heart_rate_labels = torch.from_numpy(heart_rate_labels).float()

        n_samples = radar_data.shape[0]
        print(f"Data shapes: radar={radar_data.shape}, ECG={ecg_data.shape}, HR={heart_rate_labels.shape}")
        if n_samples < 2:
            print("Not enough samples for training")
            return

        permutation = torch.randperm(n_samples)
        n_train = max(1, int(n_samples * 0.8))
        n_train = min(n_train, n_samples - 1)
        train_indices = permutation[:n_train]
        val_indices = permutation[n_train:]

        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_hr_losses = []
            epoch_feature_losses = []
            epoch_attention_losses = []
            shuffled_indices = train_indices[torch.randperm(len(train_indices))]

            for i in range(0, len(shuffled_indices), batch_size):
                batch_idx = shuffled_indices[i:i + batch_size]
                metrics = self.train_step(radar_data[batch_idx], ecg_data[batch_idx], heart_rate_labels[batch_idx])
                epoch_losses.append(metrics['total_loss'])
                epoch_hr_losses.append(metrics['hr_loss'])
                epoch_feature_losses.append(metrics['feature_loss'])
                if metrics['attention_loss'] > 0:
                    epoch_attention_losses.append(metrics['attention_loss'])

            val_batch_idx = val_indices[:min(100, len(val_indices))]
            val_metrics = self.validate(radar_data[val_batch_idx], ecg_data[val_batch_idx], heart_rate_labels[val_batch_idx])
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            avg_hr_loss = np.mean(epoch_hr_losses) if epoch_hr_losses else 0
            avg_feature_loss = np.mean(epoch_feature_losses) if epoch_feature_losses else 0
            avg_attention_loss = np.mean(epoch_attention_losses) if epoch_attention_losses else 0
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f} "
                  f"(HR: {avg_hr_loss:.4f}, Feat: {avg_feature_loss:.4f}, Attn: {avg_attention_loss:.4f}) | "
                  f"Val MAE: {val_metrics['mae']:.2f} BPM, RMSE: {val_metrics['rmse']:.2f} BPM")

        print("Training complete")
        self.save_student_model()
    
    def _generate_synthetic_data(self, n_samples: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """鐢熸垚鍚堟垚鏁版嵁"""
        ecg_data = np.random.randn(n_samples, seq_len, 128).astype(np.float32)
        radar_data = np.random.randn(n_samples, seq_len, 64).astype(np.float32)
        heart_rate_labels = np.random.rand(n_samples) * 140 + 40
        
        return radar_data, ecg_data, heart_rate_labels

    def _make_sequence_windows(self, radar_features: np.ndarray, ecg_features: np.ndarray,
                               hr_labels: np.ndarray, seq_len: int = 32,
                               stride: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert aligned frame-level features to Transformer sequence samples."""
        n_frames = min(len(radar_features), len(ecg_features), len(hr_labels))
        if n_frames == 0:
            raise ValueError("No aligned frames available for sequence windows")

        radar_features = radar_features[:n_frames]
        ecg_features = ecg_features[:n_frames]
        hr_labels = hr_labels[:n_frames]

        if n_frames < seq_len:
            pad_count = seq_len - n_frames
            radar_features = np.vstack([radar_features, np.repeat(radar_features[-1:], pad_count, axis=0)])
            ecg_features = np.vstack([ecg_features, np.repeat(ecg_features[-1:], pad_count, axis=0)])
            hr_labels = np.concatenate([hr_labels, np.repeat(hr_labels[-1:], pad_count)])
            n_frames = seq_len

        radar_windows = []
        ecg_windows = []
        label_windows = []
        for start in range(0, n_frames - seq_len + 1, stride):
            end = start + seq_len
            radar_windows.append(radar_features[start:end])
            ecg_windows.append(ecg_features[start:end])
            label_windows.append(float(np.mean(hr_labels[start:end])))

        if not radar_windows:
            raise ValueError("Failed to create sequence windows")

        return (
            np.array(radar_windows, dtype=np.float32),
            np.array(ecg_windows, dtype=np.float32),
            np.array(label_windows, dtype=np.float32)
        )
    
    def _load_real_data(self, data_pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """鍔犺浇鐪熷疄鏁版嵁"""
        all_radar_features = []
        all_ecg_features = []
        all_hr_labels = []
        
        for pair_idx, (radar_path, ecg_path) in enumerate(data_pairs):
            try:
                print(f"\n澶勭悊绗?{pair_idx+1} 缁勬暟鎹?..")
                
                # 鍔犺浇闆疯揪鏁版嵁
                radar_frames, config = DataLoader.load_radar_data(radar_path)
                print(f"  闆疯揪甯ф暟: {len(radar_frames)}")
                
                # 鍔犺浇ECG鏁版嵁
                timestamps, adc_values, fs_ecg, hr_values = DataLoader.load_ecg_data(ecg_path)
                print(f"  ECG閲囨牱鐜? {fs_ecg:.1f} Hz")
                
                # 鐗瑰緛鎻愬彇
                radar_features = DataLoader.extract_radar_features(radar_frames, config)
                radar_timestamps = DataLoader.build_radar_timestamps(radar_path, config, radar_features.shape[0])
                
                # 瀵归綈鏁版嵁
                radar_aligned, ecg_aligned, hr_labels = DataLoader.align_and_pair_by_time(
                    radar_features, radar_timestamps, timestamps, adc_values, fs_ecg, hr_values
                )
                print(f"  HR labels: mean={np.mean(hr_labels):.1f} BPM, range=({np.min(hr_labels):.1f}, {np.max(hr_labels):.1f})")
                radar_aligned, ecg_aligned, hr_labels = self._make_sequence_windows(
                    radar_aligned, ecg_aligned, hr_labels, seq_len=32, stride=4
                )
                print(f"  Sequence windows: {radar_aligned.shape[0]} samples, seq_len={radar_aligned.shape[1]}")
                
                all_radar_features.append(radar_aligned)
                all_ecg_features.append(ecg_aligned)
                all_hr_labels.append(hr_labels)
                
                print(f"  [OK] Pair {pair_idx + 1} processed")
                
            except Exception as e:
                print(f"  [ERROR] Pair {pair_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_radar_features:
            radar_data = np.concatenate(all_radar_features, axis=0).astype(np.float32)
            ecg_data = np.concatenate(all_ecg_features, axis=0).astype(np.float32)
            hr_labels = np.concatenate(all_hr_labels).astype(np.float32)
            
            print(f"\nTotal samples: {len(radar_data)}")
            return radar_data, ecg_data, hr_labels
        else:
            print("No data loaded successfully")
            return None, None, None


class SimpleRadarTransformer(nn.Module):
    """鏇寸畝鍖栫殑Radar Transformer鐢ㄤ簬瀹為檯閮ㄧ讲"""
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, 
                                       dim_feedforward=256, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """杈撳叆: (batch, seq_len, features) -> 杈撳嚭: (batch, 1)"""
        x_proj = self.input_proj(x)
        seq_len = x_proj.size(1)
        
        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        x_proj = x_proj + (position / seq_len)
        
        # Transformer缂栫爜
        encoded = self.transformer(x_proj)
        
        # 鍏ㄥ眬骞冲潎姹犲寲
        pooled = encoded.mean(dim=1)
        
        # 蹇冪巼棰勬祴
        heart_rate = self.regressor(pooled)
        
        return heart_rate
    
    def export_onnx(self, path: str = 'heart_rate_student_simple.onnx'):
        """瀵煎嚭涓篛NNX鏍煎紡"""
        self.eval()
        dummy_input = torch.randn(1, 100, 64)
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=['radar_input'],
            output_names=['heart_rate'],
            opset_version=12
        )
        print(f"绠€鍖栨ā鍨嬪凡瀵煎嚭涓? {path}")


def main():
    print("=" * 60)
    print("Cross-modal KD training")
    print("ECG Transformer -> Feature KD -> Radar Transformer -> Heart Rate")
    print("=" * 60)
    print("\nData source:")
    print("1. Load real/simulated files (radar.npy + ECGLog txt)")
    print("2. Use random synthetic demo data")
    choice = input("\nSelect (1/2): ").strip()
    use_real_data = choice == "1"

    trainer = KDTrainer(device="cpu")
    try:
        trainer.train(num_epochs=30, batch_size=16, use_real_data=use_real_data)
        print("\nModel files:")
        print("  - heart_rate_student.pt")
        print("  - heart_rate_student.npz")
        print("  - heart_rate_student.onnx (if ONNX export dependencies are installed)")
    except Exception as exc:
        print(f"Training failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        print(f"PyTorch version: {torch.__version__}")
        main()
    except ImportError:
        print("PyTorch is not installed. Install dependencies with: pip install -r requirements.txt")

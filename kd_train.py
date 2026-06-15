"""
简化版跨模态蒸馏训练程序
ECG Transformer → Feature KD → Radar Transformer → Heart Rate
总损失: L = L_HR + 0.5*L_feature + 0.2*L_attention
支持真实数据加载: radar.npy (BGT60TR13C) + ECGLog-*.txt
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

    """加载和处理真实的雷达和ECG数据"""
    
    @staticmethod
    def load_radar_data(radar_npy_path: str) -> Tuple[np.ndarray, Dict]:
        """
        加载BGT60TR13C雷达数据 (radar.npy)
        返回: (radar_frames, config)
        radar_frames shape: (n_frames, n_rx, n_chirps, n_samples)
        """
        if not os.path.exists(radar_npy_path):
            raise FileNotFoundError(f"雷达文件不存在: {radar_npy_path}")
        
        radar_frames = np.load(radar_npy_path)
        print(f"  加载雷达数据: {radar_frames.shape}")
        
        # 加载配置
        config_path = os.path.join(os.path.dirname(radar_npy_path), 'config.json')
        config = DataLoader._load_json_if_exists(config_path)
        
        return radar_frames, config
    
    @staticmethod
    def load_ecg_data(ecg_txt_path: str) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        加载ECGLog-*.txt 文件
        返回: (timestamps, adc_values, fs_ecg)
        ECG 格式: timestamp ADC HR_4s HR_30s
        """
        if not os.path.exists(ecg_txt_path):
            raise FileNotFoundError(f"ECG文件不存在: {ecg_txt_path}")
        
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
        
        # 估计采样率（从时间戳差异）
        if len(timestamps) > 1:
            positive_diffs = np.diff(timestamps)
            positive_diffs = positive_diffs[positive_diffs > 0]
            dt = np.median(positive_diffs) if len(positive_diffs) else 0
            fs_ecg = 1.0 / dt if dt > 0 else 250.0
        else:
            fs_ecg = 250.0
        
        print(f"  加载ECG数据: {len(adc_values)} 个样本, fs={fs_ecg:.1f} Hz")
        
        valid_hr_count = int(np.sum(np.isfinite(hr_values)))
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        print(f"  ECG duration={duration:.2f}s, valid HR labels={valid_hr_count}")
        return timestamps, adc_values, fs_ecg, hr_values
    
    @staticmethod
    def extract_radar_features(radar_frames: np.ndarray, config: Dict) -> np.ndarray:
        """
        从雷达帧中提取特征
        输入: (n_frames, n_rx, n_chirps, n_samples)
        输出: (n_frames, feature_dim) - 64 维特征
        """
        n_frames = radar_frames.shape[0]
        
        # 简化的特征提取：对每一帧进行 FFT，提取能量特征
        features = []
        
        for frame_idx in range(n_frames):
            frame = radar_frames[frame_idx]  # (n_rx, n_chirps, n_samples)
            
            # 对所有天线和扫频进行 FFT
            frame_fft = np.abs(np.fft.fft(frame, axis=-1))
            
            # 提取多个能量点作为特征
            # 从距离维度取几个关键频率的能量
            n_freqs = min(64, frame_fft.shape[-1] // 2)
            
            # 聚合所有RX和chirps的能量
            energy = np.mean(frame_fft[:, :, :n_freqs], axis=(0, 1))  # (n_freqs,)
            
            # 如果维度不足64，进行插值
            if len(energy) < 64:
                energy = np.interp(np.linspace(0, len(energy)-1, 64), 
                                 np.arange(len(energy)), energy)
            else:
                energy = energy[:64]
            
            features.append(energy)
        
        features = np.array(features, dtype=np.float32)
        return features
    
    @staticmethod
    def extract_ecg_features(adc_values: np.ndarray, fs_ecg: float, 
                            target_length: int = 100) -> np.ndarray:
        """
        从ECG数据中提取特征
        使用带通滤波 (0.5-40 Hz) 和下采样
        输入: (n_samples,)
        输出: (target_length, 128)
        """
        # 带通滤波 0.5-40 Hz
        try:
            b, a = butter(4, [0.5, 40.0], btype='band', fs=fs_ecg)
            ecg_filtered = filtfilt(b, a, adc_values)
        except Exception:
            ecg_filtered = adc_values
        
        # 去直流
        ecg_filtered = ecg_filtered - np.mean(ecg_filtered)
        
        # 标准化
        std = np.std(ecg_filtered)
        if std > 0:
            ecg_filtered = ecg_filtered / std
        
        # 分窗提取特征 (使用短时能量和过零率等)
        window_size = int(fs_ecg * 0.5)  # 0.5秒窗口
        hop_size = window_size // 2
        
        features = []
        for i in range(0, len(ecg_filtered) - window_size, hop_size):
            window = ecg_filtered[i:i+window_size]
            
            # 提取128维特征
            feat = np.zeros(128, dtype=np.float32)
            
            # 前32维：窗口内直接采样
            feat[:32] = np.interp(np.linspace(0, window_size-1, 32),
                                 np.arange(window_size), window)
            
            # 后32维：FFT 频域特征
            fft_mag = np.abs(np.fft.rfft(window))[:32]
            feat[32:64] = np.interp(np.linspace(0, len(fft_mag)-1, 32),
                                    np.arange(len(fft_mag)), fft_mag)
            
            # 64~96维：能量特征（多个频段）
            feat[64:96] = np.convolve(window, np.ones(5)/5, mode='same')[:32]
            
            # 96~128维：导数特征
            feat[96:128] = np.convolve(np.diff(window), np.ones(3)/3, mode='same')[:32]
            
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        
        # 调整到目标长度
        if len(features) < target_length:
            # 补零
            features = np.vstack([
                features,
                np.zeros((target_length - len(features), 128), dtype=np.float32)
            ])
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
        对齐雷达和ECG特征到相同长度
        返回: (aligned_radar_features, aligned_ecg_features)
        """
        radar_len = radar_features.shape[0]
        ecg_len = ecg_features.shape[0]
        
        # 最小长度
        min_len = min(radar_len, ecg_len)
        
        aligned_radar = radar_features[:min_len]
        aligned_ecg = ecg_features[:min_len]
        
        print(f"  对齐后: 雷达 {aligned_radar.shape}, ECG {aligned_ecg.shape}")
        
        return aligned_radar, aligned_ecg


class DataSelector:
    """UI 文件选择器"""
    
    @staticmethod
    def select_data_pairs() -> List[Tuple[str, str]]:
        """
        让用户选择多组(雷达文件, ECG文件)
        返回: [(radar_path, ecg_path), ...]
        """
        pairs = []
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        print("\n" + "="*60)
        print("数据文件选择")
        print("="*60)
        
        pair_idx = 1
        while True:
            print(f"\n第 {pair_idx} 组:")
            
            # 选择雷达文件
            print("  请选择 radar.npy 文件...")
            radar_path = filedialog.askopenfilename(
                title=f"选择第 {pair_idx} 组的 radar.npy 文件",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            
            if not radar_path:
                break
            
            # 选择ECG文件
            print(f"  请选择对应的 ECGLog-*.txt 文件...")
            ecg_path = filedialog.askopenfilename(
                title=f"选择第 {pair_idx} 组的 ECG 文件",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if not ecg_path:
                print("  ECG文件选择已取消")
                break
            
            pairs.append((radar_path, ecg_path))
            print(f"  ✓ 已添加第 {pair_idx} 组数据")
            print(f"    雷达: {os.path.basename(os.path.dirname(radar_path))}/radar.npy")
            print(f"    ECG: {os.path.basename(ecg_path)}")
            
            pair_idx += 1
            
            # 询问是否继续
            response = input("\n是否继续选择下一组数据? (y/n): ").strip().lower()
            if response != 'y':
                break
        
        root.destroy()
        
        if pairs:
            print(f"\n✓ 已选择 {len(pairs)} 组数据")
        else:
            print("\n✗ 未选择任何数据")
        
        return pairs


class ECGTransformer(nn.Module):
    """ECG Transformer教师模型 (简化版)"""
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
        """返回预测值和特征"""
        # x shape: (batch, seq_len, input_dim)
        x_proj = self.input_proj(x)
        
        # 生成位置编码
        seq_len = x_proj.size(1)
        pos_encoding = self._positional_encoding(seq_len, x_proj.size(-1)).to(x.device)
        x_proj = x_proj + pos_encoding
        
        transformer_output = self.transformer(x_proj)
        features = transformer_output.mean(dim=1)  # 全局池化
        heart_rate = self.reg_head(features)
        
        return heart_rate, features
    
    def _positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """生成位置编码 (简化版)"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(1, seq_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_encoding
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """获取指定层的注意力图 (用于注意力蒸馏)"""
        x_proj = self.input_proj(x)
        seq_len = x_proj.size(1)
        pos_encoding = self._positional_encoding(seq_len, x_proj.size(-1)).to(x.device)
        x_proj = x_proj + pos_encoding
        
        # 保存注意力图
        attention_maps = []
        def hook_fn(module, input, output):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attn_map'):
                attention_maps.append(module.self_attn.attn_map)
        
        # 注册钩子
        hooks = []
        for i, layer in enumerate(self.transformer.layers):
            if layer_idx == -1 or i == layer_idx:
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
        
        # 前向传播
        with torch.no_grad():
            _ = self.transformer(x_proj)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_maps[0] if attention_maps else None


class RadarTransformer(nn.Module):
    """Radar Transformer学生模型 (简化版)"""
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
        self.feature_proj = nn.Linear(hidden_dim, 256)  # 投影到教师特征维度
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回预测值和投影后的特征"""
        # x shape: (batch, seq_len, input_dim)
        x_proj = self.input_proj(x)
        
        # 简单位置编码
        seq_len = x_proj.size(1)
        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        x_proj = x_proj + (position / seq_len)
        
        transformer_output = self.transformer(x_proj)
        features = transformer_output.mean(dim=1)  # 全局池化
        heart_rate = self.reg_head(features)
        
        # 投影特征用于蒸馏
        projected_features = self.feature_proj(features)
        
        return heart_rate, projected_features
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """获取指定层的注意力图 (用于注意力蒸馏)"""
        x_proj = self.input_proj(x)
        seq_len = x_proj.size(1)
        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        x_proj = x_proj + (position / seq_len)
        
        # 保存注意力图
        attention_maps = []
        def hook_fn(module, input, output):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attn_map'):
                attention_maps.append(module.self_attn.attn_map)
        
        # 注册钩子
        hooks = []
        for i, layer in enumerate(self.transformer.layers):
            if layer_idx == -1 or i == layer_idx:
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
        
        # 前向传播
        with torch.no_grad():
            _ = self.transformer(x_proj)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_maps[0] if attention_maps else None


class KDTrainer:
    """跨模态蒸馏训练器"""
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 创建教师和学生模型
        self.teacher = ECGTransformer(input_dim=128, hidden_dim=256, nhead=8, num_layers=4).to(self.device)
        self.student = RadarTransformer(input_dim=64, hidden_dim=128, nhead=4, num_layers=2).to(self.device)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.feature_loss = nn.MSELoss()  # 特征蒸馏损失
        self.attention_loss = nn.MSELoss()  # 注意力蒸馏损失
        
        # 优化器
        self.optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        
        # 损失权重
        self.lambda_hr = 1.0
        self.lambda_feature = 0.5
        self.lambda_attention = 0.2
        
    def train_step(self, radar_data: torch.Tensor, ecg_data: torch.Tensor, 
                   heart_rate_labels: torch.Tensor) -> dict:
        """单步训练"""
        # 移动到设备
        radar_data = radar_data.to(self.device)
        ecg_data = ecg_data.to(self.device)
        heart_rate_labels = heart_rate_labels.to(self.device)
        
        # 教师前向传播 (冻结参数)
        with torch.no_grad():
            teacher_hr, teacher_features = self.teacher(ecg_data)
            teacher_attention = self.teacher.get_attention_maps(ecg_data)
        
        # 学生前向传播
        student_hr, student_features = self.student(radar_data)
        student_attention = self.student.get_attention_maps(radar_data)
        
        # 计算各项损失
        hr_loss = self.mse_loss(student_hr.squeeze(), heart_rate_labels)
        
        # 特征蒸馏损失
        feature_loss = self.feature_loss(student_features, teacher_features.detach())
        
        # 注意力蒸馏损失
        attention_loss = 0.0
        if teacher_attention is not None and student_attention is not None:
            # 对注意力图进行平均
            teacher_attn_mean = teacher_attention.mean(dim=1)  # 平均多头注意力
            student_attn_mean = student_attention.mean(dim=1)
            attention_loss = self.attention_loss(student_attn_mean, teacher_attn_mean.detach())
        
        # 总损失
        total_loss = (self.lambda_hr * hr_loss + 
                     self.lambda_feature * feature_loss + 
                     self.lambda_attention * attention_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'hr_loss': hr_loss.item(),
            'feature_loss': feature_loss.item(),
            'attention_loss': attention_loss.item() if teacher_attention is not None else 0.0,
            'predicted_hr': student_hr.detach().cpu().numpy(),
            'true_hr': heart_rate_labels.cpu().numpy()
        }
    
    def validate(self, radar_data: torch.Tensor, ecg_data: torch.Tensor,
                 heart_rate_labels: torch.Tensor) -> dict:
        """验证"""
        self.student.eval()
        with torch.no_grad():
            radar_data = radar_data.to(self.device)
            student_hr, _ = self.student(radar_data)
            
            hr_loss = self.mse_loss(student_hr.squeeze(), heart_rate_labels.to(self.device))
            
            # 计算心率预测误差
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
        """保存学生模型为ONNX格式"""
        self.student.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 100, 64).to(self.device)
        
        # 导出ONNX
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
        print(f"学生模型已保存为: {path}")
        
        # 同时保存为PyTorch格式
        torch.save(self.student.state_dict(), 'heart_rate_student.pt')
        print("学生模型权重已保存为: heart_rate_student.pt")
        
        # 保存为NPZ格式
        npz_path = 'heart_rate_student.npz'
        state_dict = self.student.state_dict()
        npz_dict = {}
        for key, value in state_dict.items():
            npz_dict[key] = value.cpu().numpy()
        np.savez(npz_path, **npz_dict)
        print(f"学生模型权重已保存为: {npz_path}")
        
        self.student.train()
    
    def train(self, num_epochs: int = 50, batch_size: int = 32, use_real_data: bool = True):
        """
        完整训练流程
        支持真实数据或模拟数据
        """
        print("开始训练跨模态蒸馏模型...")
        print(f"教师模型: ECG Transformer ({sum(p.numel() for p in self.teacher.parameters()):,} 参数)")
        print(f"学生模型: Radar Transformer ({sum(p.numel() for p in self.student.parameters()):,} 参数)")
        print(f"损失权重: HR={self.lambda_hr}, Feature={self.lambda_feature}, Attention={self.lambda_attention}")
        
        # 数据加载
        if use_real_data:
            print("\n加载真实数据...")
            data_pairs = DataSelector.select_data_pairs()
            
            if not data_pairs:
                print("未选择任何数据，使用模拟数据代替")
                radar_data, ecg_data, heart_rate_labels = self._generate_synthetic_data(1000, 100)
            else:
                radar_data, ecg_data, heart_rate_labels = self._load_real_data(data_pairs)
        else:
            print("生成模拟数据用于训练演示...")
            radar_data, ecg_data, heart_rate_labels = self._generate_synthetic_data(1000, 100)
        
        if radar_data is None or len(radar_data) == 0:
            print("数据加载失败")
            return
        
        radar_data = torch.from_numpy(radar_data).float()
        ecg_data = torch.from_numpy(ecg_data).float()
        heart_rate_labels = torch.from_numpy(heart_rate_labels).float()
        
        n_samples = radar_data.shape[0]
        print(f"数据形状: 雷达{radar_data.shape}, ECG{ecg_data.shape}, HR标签{heart_rate_labels.shape}")
        
        # 数据划分
        train_ratio = 0.8
        n_train = int(n_samples * train_ratio)
        
        train_indices = torch.randperm(n_samples)[:n_train]
        val_indices = torch.randperm(n_samples)[n_train:]
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_hr_losses = []
            epoch_feature_losses = []
            epoch_attention_losses = []
            
            # 随机打乱训练数据
            shuffled_indices = torch.randperm(len(train_indices))
            
            for i in range(0, len(train_indices), batch_size):
                batch_idx = train_indices[shuffled_indices[i:i+batch_size]]
                
                batch_radar = radar_data[batch_idx]
                batch_ecg = ecg_data[batch_idx]
                batch_labels = heart_rate_labels[batch_idx]
                
                # 训练步
                metrics = self.train_step(batch_radar, batch_ecg, batch_labels)
                
                epoch_losses.append(metrics['total_loss'])
                epoch_hr_losses.append(metrics['hr_loss'])
                epoch_feature_losses.append(metrics['feature_loss'])
                if metrics['attention_loss'] > 0:
                    epoch_attention_losses.append(metrics['attention_loss'])
            
            # 验证
            val_batch_idx = val_indices[:min(100, len(val_indices))]
            val_metrics = self.validate(
                radar_data[val_batch_idx],
                ecg_data[val_batch_idx],
                heart_rate_labels[val_batch_idx]
            )
            
            # 打印进度
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            avg_hr_loss = np.mean(epoch_hr_losses) if epoch_hr_losses else 0
            avg_feature_loss = np.mean(epoch_feature_losses) if epoch_feature_losses else 0
            avg_attention_loss = np.mean(epoch_attention_losses) if epoch_attention_losses else 0
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} (HR: {avg_hr_loss:.4f}, "
                  f"Feat: {avg_feature_loss:.4f}, Attn: {avg_attention_loss:.4f}) | "
                  f"Val MAE: {val_metrics['mae']:.2f} BPM, RMSE: {val_metrics['rmse']:.2f} BPM")
        
        print("训练完成!")
        
        # 保存模型
        self.save_student_model()
    
    def _generate_synthetic_data(self, n_samples: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成合成数据"""
        ecg_data = np.random.randn(n_samples, seq_len, 128).astype(np.float32)
        radar_data = np.random.randn(n_samples, seq_len, 64).astype(np.float32)
        heart_rate_labels = np.random.rand(n_samples) * 140 + 40
        
        return radar_data, ecg_data, heart_rate_labels
    
    def _load_real_data(self, data_pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载真实数据"""
        all_radar_features = []
        all_ecg_features = []
        all_hr_labels = []
        
        for pair_idx, (radar_path, ecg_path) in enumerate(data_pairs):
            try:
                print(f"\n处理第 {pair_idx+1} 组数据...")
                
                # 加载雷达数据
                radar_frames, config = DataLoader.load_radar_data(radar_path)
                print(f"  雷达帧数: {len(radar_frames)}")
                
                # 加载ECG数据
                timestamps, adc_values, fs_ecg, hr_values = DataLoader.load_ecg_data(ecg_path)
                print(f"  ECG采样率: {fs_ecg:.1f} Hz")
                
                # 特征提取
                radar_features = DataLoader.extract_radar_features(radar_frames, config)
                radar_timestamps = DataLoader.build_radar_timestamps(radar_path, config, radar_features.shape[0])
                
                # 对齐数据
                radar_aligned, ecg_aligned, hr_labels = DataLoader.align_and_pair_by_time(
                    radar_features, radar_timestamps, timestamps, adc_values, fs_ecg, hr_values
                )
                print(f"  HR labels: mean={np.mean(hr_labels):.1f} BPM, range=({np.min(hr_labels):.1f}, {np.max(hr_labels):.1f})")
                
                all_radar_features.append(radar_aligned)
                all_ecg_features.append(ecg_aligned)
                all_hr_labels.append(hr_labels)
                
                print(f"  ✓ 第 {pair_idx+1} 组数据处理完成")
                
            except Exception as e:
                print(f"  ✗ 第 {pair_idx+1} 组数据处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 合并所有数据
        if all_radar_features:
            radar_data = np.vstack(all_radar_features).astype(np.float32)
            ecg_data = np.vstack(all_ecg_features).astype(np.float32)
            hr_labels = np.concatenate(all_hr_labels).astype(np.float32)
            
            print(f"\n总数据量: {len(radar_data)} 个样本")
            return radar_data, ecg_data, hr_labels
        else:
            print("未成功加载任何数据")
            return None, None, None


class SimpleRadarTransformer(nn.Module):
    """更简化的Radar Transformer用于实际部署"""
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
        """输入: (batch, seq_len, features) -> 输出: (batch, 1)"""
        x_proj = self.input_proj(x)
        seq_len = x_proj.size(1)
        
        # 简单位置编码
        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        x_proj = x_proj + (position / seq_len)
        
        # Transformer编码
        encoded = self.transformer(x_proj)
        
        # 全局平均池化
        pooled = encoded.mean(dim=1)
        
        # 心率预测
        heart_rate = self.regressor(pooled)
        
        return heart_rate
    
    def export_onnx(self, path: str = 'heart_rate_student_simple.onnx'):
        """导出为ONNX格式"""
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
        print(f"简化模型已导出为: {path}")


def main():
    """主函数"""
    print("=" * 60)
    print("跨模态蒸馏训练程序")
    print("架构: ECG Transformer → Feature KD → Radar Transformer → Heart Rate")
    print("损失: L = L_HR + 0.5*L_feature + 0.2*L_attention")
    print("数据源: 真实BGT60TR13C雷达 + ECG日志")
    print("=" * 60)
    
    # 用户选择数据来源
    print("\n数据源选择:")
    print("1. 加载真实数据 (radar.npy + ECGLog-*.txt)")
    print("2. 使用模拟数据进行演示训练")
    
    choice = input("\n请选择 (1/2): ").strip()
    use_real_data = choice == '1'
    
    # 创建训练器
    trainer = KDTrainer(device='cpu')  # 使用CPU以兼容更多环境
    
    # 开始训练
    try:
        if use_real_data:
            print("\n开始从真实文件加载数据...")
            trainer.train(num_epochs=30, batch_size=16, use_real_data=True)
        else:
            print("\n开始生成模拟数据并训练...")
            trainer.train(num_epochs=30, batch_size=16, use_real_data=False)
        
        print("\n" + "=" * 60)
        print("模型导出:")
        print("=" * 60)
        print("已生成以下模型文件:")
        print("  - heart_rate_student.onnx (ONNX格式)")
        print("  - heart_rate_student.pt (PyTorch权重)")
        print("  - heart_rate_student.npz (NumPy权重)")
        print("\n在heart_rate_monitor.py中使用:")
        print("1. 加载ONNX模型进行推理")
        print("2. 或加载NumPy权重重新构建模型")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 创建简化模型备用
        print("\n创建备用简化模型...")
        simple_model = SimpleRadarTransformer()
        simple_model.export_onnx()
        print("已创建简化模型: heart_rate_student_simple.onnx")


def create_minimal_training_example():
    """创建最小化训练示例，用于演示和测试"""
    print("创建最小化训练示例...")
    
    # 超小模型用于演示
    class TinyECGTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            self.regressor = nn.Linear(32, 1)
        
        def forward(self, x):
            features = self.encoder(x.mean(dim=1))  # 简化：先平均再编码
            hr = self.regressor(features)
            return hr, features
    
    class TinyRadarStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
            self.feature_proj = nn.Linear(16, 32)  # 投影到教师特征维度
            self.regressor = nn.Linear(16, 1)
        
        def forward(self, x):
            features = self.encoder(x.mean(dim=1))
            hr = self.regressor(features)
            projected = self.feature_proj(features)
            return hr, projected
    
    # 训练示例
    teacher = TinyECGTeacher()
    student = TinyRadarStudent()
    
    # 模拟数据
    ecg_data = torch.randn(10, 100, 128)
    radar_data = torch.randn(10, 100, 64)
    labels = torch.rand(10) * 140 + 40
    
    # 教师预测
    with torch.no_grad():
        teacher_hr, teacher_features = teacher(ecg_data)
    
    # 学生训练
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    for epoch in range(10):
        student_hr, student_features = student(radar_data)
        
        hr_loss = mse_loss(student_hr.squeeze(), labels)
        feature_loss = mse_loss(student_features, teacher_features.detach())
        
        total_loss = hr_loss + 0.5 * feature_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss.item():.4f} "
              f"(HR: {hr_loss.item():.4f}, Feature: {feature_loss.item():.4f})")
    
    # 保存学生模型
    torch.save(student.state_dict(), 'tiny_student.pt')
    print("最小化示例完成，已保存为: tiny_student.pt")


if __name__ == "__main__":
    # 检查PyTorch是否可用
    try:
        import torch
        import torch.nn as nn
        print(f"PyTorch版本: {torch.__version__}")
        main()
    except ImportError:
        print("PyTorch未安装，正在安装...")
        print("请运行: pip install torch torchvision")
        print("或: pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html")
        
        # 创建简单的训练说明文件
        with open('training_instructions.txt', 'w') as f:
            f.write("""跨模态蒸馏训练程序 - 安装和使用说明

1. 安装依赖:
pip install torch torchvision numpy

2. 运行训练:
python kd_train.py

3. 将生成以下模型文件:
   - heart_rate_student.onnx
   - heart_rate_student.pt
   - heart_rate_student.npz

4. 在heart_rate_monitor.py中使用:
   import onnxruntime
   import numpy as np
   
   # 加载ONNX模型
   session = onnxruntime.InferenceSession('heart_rate_student.onnx')
   
   # 准备输入数据 (批大小, 序列长度, 特征维度)
   radar_input = np.random.randn(1, 100, 64).astype(np.float32)
   
   # 推理
   input_name = session.get_inputs()[0].name
   output_name = session.get_outputs()[0].name
   heart_rate = session.run([output_name], {input_name: radar_input})[0]
   
   print(f"预测心率: {heart_rate[0,0]:.1f} BPM")

5. 论文方法概述:
   ECG Transformer → Feature KD → Radar Transformer → Heart Rate
   总损失: L = L_HR + 0.5*L_feature + 0.2*L_attention
            """)
        print("已创建训练说明文件: training_instructions.txt")

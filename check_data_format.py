"""
数据格式检查脚本
验证 radar.npy 和 ECGLog-*.txt 的数据格式是否与 kd_train.py 兼容
"""

import json
import numpy as np
import os

def check_radar_format():
    """检查雷达数据格式"""
    print("\n" + "="*60)
    print("检查雷达数据格式")
    print("="*60)
    
    radar_path = r"e:\GitRepository\FMCWControl\bgtr13cTest\BGT60TR13C_record_20260604-161236\RadarIfxAvian_00\radar.npy"
    
    if not os.path.exists(radar_path):
        print(f"✗ 雷达文件不存在: {radar_path}")
        return
    
    try:
        # 加载雷达数据
        radar_data = np.load(radar_path)
        print(f"✓ 成功加载雷达文件")
        print(f"  形状: {radar_data.shape}")
        print(f"  数据类型: {radar_data.dtype}")
        print(f"  内存大小: {radar_data.nbytes / 1024 / 1024:.1f} MB")
        
        # 解析配置
        config_path = os.path.join(os.path.dirname(radar_path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"\n配置信息:")
            
            device_cfg = config.get('device_config', {}).get('fmcw_single_shape', {})
            print(f"  采样率: {device_cfg.get('sample_rate_Hz', 'N/A')} Hz")
            print(f"  每帧扫频数: {device_cfg.get('num_chirps_per_frame', 'N/A')}")
            print(f"  每扫频采样数: {device_cfg.get('num_samples_per_chirp', 'N/A')}")
            print(f"  RX天线: {device_cfg.get('rx_antennas', 'N/A')}")
            print(f"  TX天线: {device_cfg.get('tx_antennas', 'N/A')}")
            print(f"  帧重复时间: {device_cfg.get('frame_repetition_time_s', 'N/A')} s")
        
        # 解析元数据
        meta_path = os.path.join(os.path.dirname(radar_path), 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"\n元数据:")
            print(f"  SDK版本: {meta.get('sdk_version', 'N/A')}")
            print(f"  固件版本: {meta.get('firmware_version', 'N/A')}")
        
        # 数据检查
        print(f"\n数据检查:")
        print(f"  最小值: {radar_data.min():.4f}")
        print(f"  最大值: {radar_data.max():.4f}")
        print(f"  平均值: {radar_data.mean():.4f}")
        print(f"  标准差: {radar_data.std():.4f}")
        
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()


def check_ecg_format():
    """检查ECG数据格式"""
    print("\n" + "="*60)
    print("检查ECG数据格式")
    print("="*60)
    
    ecg_path = r"e:\GitRepository\FMCWControl\ECGLog-2026-4-9-12-41-24.txt"
    
    if not os.path.exists(ecg_path):
        print(f"✗ ECG文件不存在: {ecg_path}")
        return
    
    try:
        # 解析文件
        timestamps = []
        adc_values = []
        hr_4s = []
        hr_30s = []
        
        with open(ecg_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('timestamp'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        ts_str = parts[0].rstrip(':')
                        ts = float(ts_str)
                        adc = int(parts[1])
                        hr4 = int(parts[2]) if len(parts) > 2 else None
                        hr30 = int(parts[3]) if len(parts) > 3 else None
                        
                        timestamps.append(ts)
                        adc_values.append(adc)
                        if hr4 is not None:
                            hr_4s.append(hr4)
                        if hr30 is not None:
                            hr_30s.append(hr30)
                    except (ValueError, IndexError):
                        continue
        
        timestamps = np.array(timestamps)
        adc_values = np.array(adc_values)
        
        print(f"✓ 成功加载ECG文件")
        print(f"  总行数: {len(timestamps)}")
        print(f"  起始时间戳: {timestamps[0]:.3f}")
        print(f"  结束时间戳: {timestamps[-1]:.3f}")
        print(f"  持续时长: {(timestamps[-1] - timestamps[0]):.1f} 秒")
        
        # 估计采样率
        if len(timestamps) > 1:
            dt = np.median(np.diff(timestamps))
            fs_est = 1.0 / dt if dt > 0 else 250.0
            print(f"  估计采样率: {fs_est:.1f} Hz")
        
        # ADC值统计
        print(f"\nADC数据统计:")
        print(f"  最小值: {adc_values.min()}")
        print(f"  最大值: {adc_values.max()}")
        print(f"  平均值: {adc_values.mean():.1f}")
        print(f"  标准差: {adc_values.std():.1f}")
        
        # 心率统计
        if hr_4s:
            print(f"\n4秒平均心率: {np.mean(hr_4s):.1f} BPM (min={np.min(hr_4s)}, max={np.max(hr_4s)})")
        if hr_30s:
            print(f"30秒平均心率: {np.mean(hr_30s):.1f} BPM (min={np.min(hr_30s)}, max={np.max(hr_30s)})")
        
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()


def check_alignment():
    """检查数据对齐"""
    print("\n" + "="*60)
    print("检查雷达和ECG的时间对齐")
    print("="*60)
    
    try:
        radar_path = r"e:\GitRepository\FMCWControl\bgtr13cTest\BGT60TR13C_record_20260604-161236\RadarIfxAvian_00\radar.npy"
        ecg_path = r"e:\GitRepository\FMCWControl\ECGLog-2026-4-9-12-41-24.txt"
        
        # 获取雷达帧数
        radar_data = np.load(radar_path)
        n_frames = radar_data.shape[0]
        
        config_path = os.path.join(os.path.dirname(radar_path), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        frame_time_s = config['device_config']['fmcw_single_shape']['frame_repetition_time_s']
        total_radar_time = n_frames * frame_time_s
        
        print(f"雷达数据:")
        print(f"  帧数: {n_frames}")
        print(f"  帧时间间隔: {frame_time_s:.4f} s")
        print(f"  总时长: {total_radar_time:.1f} s")
        
        # 获取ECG采样数
        timestamps = []
        with open(ecg_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('timestamp'):
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        ts = float(parts[0].rstrip(':'))
                        timestamps.append(ts)
                    except ValueError:
                        continue
        
        if timestamps:
            timestamps = np.array(timestamps)
            total_ecg_time = timestamps[-1] - timestamps[0]
            n_ecg = len(timestamps)
            fs_ecg = 1.0 / np.median(np.diff(timestamps)) if len(timestamps) > 1 else 250.0
            
            print(f"\nECG数据:")
            print(f"  采样数: {n_ecg}")
            print(f"  采样率: {fs_ecg:.1f} Hz")
            print(f"  总时长: {total_ecg_time:.1f} s")
            
            # 对齐检查
            print(f"\n对齐检查:")
            if abs(total_radar_time - total_ecg_time) < 5:
                print(f"  ✓ 时长匹配 (差异: {abs(total_radar_time - total_ecg_time):.1f}s)")
            else:
                print(f"  ⚠ 时长不匹配 (差异: {abs(total_radar_time - total_ecg_time):.1f}s)")
                print(f"    可能需要时间戳同步或裁剪")
        
    except Exception as e:
        print(f"✗ 对齐检查失败: {e}")


def main():
    """主程序"""
    print("\n" + "="*60)
    print("数据格式检查 - kd_train.py 兼容性验证")
    print("="*60)
    
    check_radar_format()
    check_ecg_format()
    check_alignment()
    
    print("\n" + "="*60)
    print("检查完成")
    print("="*60)
    print("\n如果所有检查都通过，可以使用 kd_train.py 进行训练:")
    print("  python kd_train.py")


if __name__ == '__main__':
    main()

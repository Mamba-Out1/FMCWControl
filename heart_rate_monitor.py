# ===========================================================================
# 实时心率监测系统 - 后端服务
# ===========================================================================

import numpy as np
from collections import deque
import math
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import threading
import time
import traceback

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
        self.current_heart_rate = 0  # 当前心率BPM
        self.current_respiratory_rate = 0  # 当前呼吸率BPM
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
    
    def unwrap_phase(self, phase_buffer):
        """相位解缠绕"""
        phases = np.array(phase_buffer)
        unwrapped = np.unwrap(phases)
        return unwrapped
    
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
    
    def stop(self):
        """停止数据采集"""
        self.is_running = False
    
    def get_data(self):
        """获取当前数据用于API响应"""
        with self.lock:
            # 确保返回的数据都是Python原生类型，避免JSON序列化错误
            heart_waveform = [float(x) for x in list(self.heart_rate_buffer)[-100:]] if self.heart_rate_buffer else []
            resp_waveform = [float(x) for x in list(self.respiratory_buffer)[-100:]] if self.respiratory_buffer else []
            raw_waveform = [float(x) for x in list(self.raw_data_buffer)[-100:]] if self.raw_data_buffer else []
            
            return {
                'heart_rate_waveform': heart_waveform,
                'respiratory_waveform': resp_waveform,
                'raw_data_waveform': raw_waveform,  # 新增：原始数据波形
                'current_heart_rate': float(self.current_heart_rate),
                'current_respiratory_rate': float(self.current_respiratory_rate),
                'frame_count': int(self.frame_count),  # 新增：帧计数
                'is_running': bool(self.is_running),  # 新增：运行状态
                'error_message': str(self.error_message),  # 新增：错误信息
                'buffer_sizes': {  # 新增：缓冲区大小
                    'raw_phase': len(self.raw_phase_buffer),
                    'heart_rate': len(self.heart_rate_buffer),
                    'respiratory': len(self.respiratory_buffer),
                    'raw_data': len(self.raw_data_buffer)
                },
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

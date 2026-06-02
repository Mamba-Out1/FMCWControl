# 雷达心率/呼吸率实时监测系统

基于 BGT60TR13C 雷达的非接触式生命体征监测系统，可实时显示心率和呼吸率波形。

## 🌟 功能特性

- ✅ 实时心率监测 (48-150 BPM)
- ✅ 实时呼吸率监测 (9-30 BPM)
- ✅ 实时波形可视化
- ✅ 美观的Web界面
- ✅ 带通滤波信号处理
- ✅ FFT频域分析

## 📋 系统要求

- Python 3.8+
- BGT60TR13C 雷达传感器
- Windows 操作系统
- ifxRadarSDK 已安装

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动系统

```bash
python heart_rate_monitor.py
```

### 3. 访问界面

打开浏览器访问: http://localhost:5000

## 📊 工作原理

### 数据流程

1. **雷达采集**: 以20Hz频率采集雷达帧数据
2. **相位提取**: 从雷达数据中提取目标的相位信息
3. **信号分离**: 
   - 心率信号: 0.8-2.5 Hz带通滤波 (48-150 BPM)
   - 呼吸信号: 0.15-0.5 Hz带通滤波 (9-30 BPM)
4. **频率估计**: FFT分析计算实时BPM值
5. **可视化**: 通过Flask + Chart.js实时显示波形

### 核心算法

```python
# 相位提取
phase = np.angle(radar_data)

# 带通滤波
heart_signal = bandpass_filter(phase, 0.8, 2.5 Hz)
resp_signal = bandpass_filter(phase, 0.15, 0.5 Hz)

# FFT频率估计
fft_result = np.fft.fft(signal)
peak_freq = find_peak_in_band(fft_result)
bpm = peak_freq * 60
```

## 📁 文件结构

```
.
├── heart_rate_monitor.py   # 后端服务 (Flask + 雷达数据处理)
├── raw_data.py             # 原始雷达数据读取示例
├── templates/
│   └── index.html          # 前端界面
├── requirements.txt        # Python依赖
└── README.md              # 说明文档
```

## ⚙️ 配置参数

### 雷达配置 (heart_rate_monitor.py)

```python
frame_repetition_time_s=0.05    # 帧率: 20Hz
num_chirps=16                    # 每帧扫频数
num_samples=128                  # 每扫频采样点
start_frequency_Hz=59e9         # 起始频率: 59GHz
end_frequency_Hz=61e9           # 终止频率: 61GHz
```

### 信号处理参数

```python
heart_rate_band = [0.8, 2.5]     # 心率频带 (Hz)
respiratory_band = [0.15, 0.5]   # 呼吸频带 (Hz)
buffer_size = 500                # 波形显示点数
```

## 🔧 使用建议

### 最佳监测条件

- **距离**: 0.5-2米
- **姿势**: 保持静止，面向雷达
- **环境**: 减少周围移动物体干扰
- **位置**: 雷达应对准胸部区域

### 故障排除

**问题**: 无法连接雷达
- 检查雷达USB连接
- 确认ifxRadarSDK已正确安装
- 检查设备管理器中的雷达设备

**问题**: 心率值不稳定
- 确保受测者保持静止
- 调整雷达距离和角度
- 增加数据缓冲长度

**问题**: 波形显示异常
- 检查浏览器控制台错误
- 确认Flask服务正常运行
- 清除浏览器缓存

## 🎯 后续优化方向

1. **算法优化**
   - [ ] 自适应滤波器
   - [ ] 多目标检测
   - [ ] 机器学习优化

2. **功能扩展**
   - [ ] 数据记录与导出
   - [ ] 历史数据回放
   - [ ] 报警阈值设置
   - [ ] 多用户管理

3. **界面改进**
   - [ ] 响应式设计优化
   - [ ] 暗黑模式
   - [ ] 更多数据统计

## 📝 技术栈

- **后端**: Flask, NumPy, SciPy
- **前端**: HTML5, JavaScript, Chart.js
- **硬件**: BGT60TR13C雷达
- **SDK**: ifxRadarSDK

## 📄 许可证

本项目基于原始雷达SDK示例代码修改，遵循相应的开源许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request!

---

**开发时间**: 2026年6月
**版本**: 1.0.0

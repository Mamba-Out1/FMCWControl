## kd_train.py - 跨模态蒸馏模型训练说明

### 概述
`kd_train.py` 是用于训练跨模态蒸馏心率提取模型的程序。它支持从真实的 BGT60TR13C 雷达数据和 ECG 日志中加载数据进行训练。

### 文件格式

#### 1. 雷达数据格式
**位置**: `BGT60TR13C_record_YYYYMMDD-HHMMSS/RadarIfxAvian_00/`

**文件**:
- `radar.npy` - NumPy 二进制数组，包含原始雷达帧数据
  - 形状: `(n_frames, n_rx, n_chirps, n_samples)`
  - 数据类型: `uint16`
  - 示例: `(238, 3, 32, 128)` = 238 帧，3 个接收天线，每帧 32 次扫频，每次 128 个采样点

- `config.json` - 雷达配置参数（自动读取）
  ```json
  {
    "device_config": {
      "fmcw_single_shape": {
        "sample_rate_Hz": 2000000,
        "num_chirps_per_frame": 32,
        "num_samples_per_chirp": 128,
        "frame_repetition_time_s": 0.1613,
        "rx_antennas": [1, 2, 3],
        "tx_antennas": [1],
        ...
      }
    }
  }
  ```

- `meta.json` - 设备元数据

#### 2. ECG 数据格式
**文件**: `ECGLog-YYYY-M-D-HH-MM-SS.txt`

**格式**: 文本文件，空格分隔列

```
timestamp: ADC HeartRate4sAverage HeartRate30sAverage
1775709684.554:   -366  97  97
1775709684.554:   -159  97  97
1775709684.579:   -328  97  97
```

**列说明**:
- `timestamp` - Unix 时间戳（秒），表示采样时刻
- `ADC` - 原始心电信号值（16-bit 有符号整数）
- `HeartRate4sAverage` - 4秒平均心率（BPM）
- `HeartRate30sAverage` - 30秒平均心率（BPM）

### 使用流程

#### 第 1 步: 验证数据格式
```bash
python check_data_format.py
```

此脚本将验证：
- ✓ `radar.npy` 是否可以正确加载
- ✓ `ECGLog-*.txt` 是否格式正确
- ✓ 雷达和 ECG 数据的时间长度是否匹配
- ✓ 采样率是否在有效范围内

**预期输出示例**:
```
============================================================
检查雷达数据格式
============================================================
✓ 成功加载雷达文件
  形状: (238, 3, 32, 128)
  采样率: 2000000 Hz
  每帧扫频数: 32
  帧重复时间: 0.1613 s

============================================================
检查ECG数据格式
============================================================
✓ 成功加载ECG文件
  总行数: 57809
  采样率: 250.0 Hz
  总时长: 112.9 秒
```

#### 第 2 步: 启动训练
```bash
python kd_train.py
```

**交互式菜单**:
```
============================================================
跨模态蒸馏训练程序
架构: ECG Transformer → Feature KD → Radar Transformer → Heart Rate
损失: L = L_HR + 0.5*L_feature + 0.2*L_attention
数据源: 真实BGT60TR13C雷达 + ECG日志
============================================================

数据源选择:
1. 加载真实数据 (radar.npy + ECGLog-*.txt)
2. 使用模拟数据进行演示训练

请选择 (1/2): 1
```

**选择 1 - 加载真实数据**:
```
开始从真实文件加载数据...

第 1 组:
  请选择 radar.npy 文件...
  [文件对话框打开]
  
  请选择对应的 ECGLog-*.txt 文件...
  [文件对话框打开]
  
  ✓ 已添加第 1 组数据
    雷达: BGT60TR13C_record_20260604-161236/radar.npy
    ECG: ECGLog-2026-4-9-12-41-24.txt
```

#### 第 3 步: 训练过程
```
处理第 1 组数据...
  加载雷达数据: (238, 3, 32, 128)
  加载ECG数据: 57809 个样本, fs=250.0 Hz
  对齐后: 雷达 (238, 64), ECG (238, 128)
  ✓ 第 1 组数据处理完成

总数据量: 238 个样本

开始训练跨模态蒸馏模型...
教师模型: ECG Transformer (1,345,543 参数)
学生模型: Radar Transformer (467,265 参数)
损失权重: HR=1.0, Feature=0.5, Attention=0.2

Epoch  1/30 | Loss: 0.1234 (HR: 0.0523, Feat: 0.0456, Attn: 0.0032) | Val MAE: 18.45 BPM, RMSE: 22.34 BPM
Epoch  2/30 | Loss: 0.0945 (HR: 0.0412, Feat: 0.0361, Attn: 0.0028) | Val MAE: 15.23 BPM, RMSE: 18.92 BPM
...
```

### 数据加载机制

#### RadioLoader 类
```python
DataLoader.load_radar_data(radar_npy_path)
```
- 加载 `radar.npy` 文件
- 自动读取 `config.json` 获取配置参数
- 返回: `(radar_frames, config)`

#### ECG 加载
```python
DataLoader.load_ecg_data(ecg_txt_path)
```
- 解析 ECGLog-*.txt 文本文件
- 提取时间戳、ADC 值、心率标签
- 估计采样率（从时间戳计算）
- 返回: `(timestamps, adc_values, fs_ecg)`

#### 特征提取

**雷达特征提取** (64 维):
```python
radar_features = DataLoader.extract_radar_features(radar_frames, config)
```
- 对每一帧进行 FFT
- 提取频域能量特征
- 输出形状: `(n_frames, 64)`

**ECG 特征提取** (128 维):
```python
ecg_features = DataLoader.extract_ecg_features(adc_values, fs_ecg)
```
- 带通滤波 (0.5-40 Hz)
- 时域、频域、能量特征组合
- 输出形状: `(n_windows, 128)`

### 输出文件

训练完成后，生成以下文件:

1. **heart_rate_student.onnx** - ONNX 格式模型
   - 用于跨平台推理
   - 输入: `(batch, seq_len, 64)` - 雷达特征
   - 输出: `(batch, 1)` - 心率预测

2. **heart_rate_student.pt** - PyTorch 权重文件
   - 用于 PyTorch 环境继续训练或推理

3. **heart_rate_student.npz** - NumPy 权重包
   - 用于在 `heart_rate_monitor.py` 中加载

### 在 heart_rate_monitor.py 中使用

```python
# 自动加载已训练的 KD 学生模型
kd_processor = DistilledHeartRateExtractor(
    fs=20,
    heart_band=[0.8, 2.5],
    model_path='heart_rate_student.onnx'  # 或 'heart_rate_student.npz'
)

# 实时推理
heart_rate = kd_processor.estimate_hr(radar_signal)
```

### 数据对齐注意事项

⚠️ **重要**: 选择配对的雷达和 ECG 数据

如果雷达和 ECG 数据不是同时采集的，需要进行时间对齐：

1. **检查时间范围匹配**:
   ```bash
   python check_data_format.py
   ```

2. **手动对齐** (如需要):
   - 找到两个数据的共同时间段
   - 从 ECG 中提取对应时间段的数据

3. **或选择已配对的数据**:
   - 在同一个记录会话中采集的雷达和 ECG
   - 确保时间戳相近

### 故障排除

#### 1. 文件未找到
```
FileNotFoundError: 雷达文件不存在: ...
```
- 检查文件路径是否正确
- 确保 `radar.npy` 在 `RadarIfxAvian_00/` 目录下

#### 2. 数据不匹配
```
⚠ 时长不匹配 (差异: 74.6s)
```
- 雷达和 ECG 数据可能不是同时采集
- 选择配对的数据文件
- 或手动裁剪数据到相同时间段

#### 3. 采样率问题
```
RuntimeWarning: divide by zero encountered
```
- ECG 时间戳可能有重复
- 检查 ECGLog-*.txt 文件格式

#### 4. 内存不足
- 减少数据加载量
- 修改 batch_size 为更小的值

### 高级配置

#### 修改训练参数

编辑 `kd_train.py` 的 `main()` 函数:

```python
# 改变 epoch 数和 batch_size
trainer.train(num_epochs=50, batch_size=8, use_real_data=True)
```

#### 自定义特征提取

修改 `DataLoader` 类中的特征提取方法:

```python
@staticmethod
def extract_radar_features(radar_frames, config):
    # 自定义特征提取逻辑
    ...
```

### 参考

- 雷达数据格式: BGT60TR13C FMCW Radar
- ECG 数据来源: BMD101 心电模块
- 模型架构: Transformer + Cross-modal Knowledge Distillation

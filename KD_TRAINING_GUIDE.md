# 跨模态知识蒸馏训练指南

## 概述
本指南介绍如何使用论文方法训练ECG到Radar的跨模态知识蒸馏模型。

## 训练流程

```
ECG Transformer (教师)
        ↓
 Feature KD (特征蒸馏)
        ↓
Radar Transformer (学生)  
        ↓
Heart Rate 预测
```

## 总损失函数

```
L = L_HR + 0.5*L_feature + 0.2*L_attention

其中:
- L_HR: 心率回归损失 (MSE)
- L_feature: ECG-Radar特征蒸馏损失
- L_attention: 注意力蒸馏损失
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行训练
```bash
python kd_train.py
```

### 3. 生成的模型文件
训练完成后将生成以下文件：
- `heart_rate_student.onnx` - ONNX格式模型 (推荐)
- `heart_rate_student.pt` - PyTorch权重文件
- `heart_rate_student.npz` - NumPy权重文件

### 4. 集成到监控系统
```bash
python kd_integration.py
```

## 详细步骤

### 第1步：准备数据
创建 `data_preparation.py` 来准备训练数据：
```python
# 从雷达数据提取相位序列
# 从ECG数据提取特征序列
# 同步心率标签
```

### 第2步：配置模型参数
在 `kd_train.py` 中修改：
```python
# ECG教师模型参数
teacher_hidden_dim = 256
teacher_nhead = 8
teacher_layers = 4

# Radar学生模型参数  
student_hidden_dim = 128
student_nhead = 4
student_layers = 2

# 训练参数
batch_size = 32
learning_rate = 1e-3
num_epochs = 50
```

### 第3步：开始训练
```python
trainer = KDTrainer()
trainer.train(num_epochs=50, batch_size=32)
```

### 第4步：评估模型
```python
# 在验证集上评估
val_metrics = trainer.validate(val_radar, val_ecg, val_labels)
print(f"MAE: {val_metrics['mae']:.2f} BPM")
print(f"RMSE: {val_metrics['rmse']:.2f} BPM")
```

### 第5步：导出模型
```python
trainer.save_student_model('heart_rate_student.onnx')
```

## 监控系统集成

### 方法1：使用ONNX模型 (推荐)
```python
# 在heart_rate_monitor.py中添加
from kd_integration import KDHeartRatePredictor

class RadarDataProcessor:
    def __init__(self):
        self.kd_predictor = KDHeartRatePredictor('heart_rate_student.onnx')
        self.current_kd_hr = 0.0
        
    def process_radar_data(self):
        # 在每帧数据中
        if frame_data is not None:
            kd_hr = self.kd_predictor.predict_heart_rate(frame_data)
            if kd_hr is not None:
                self.current_kd_hr = kd_hr
```

### 方法2：使用NumPy权重
```python
predictor = KDHeartRatePredictor('heart_rate_student.npz')
```

### 方法3：使用PyTorch模型
```python
predictor = KDHeartRatePredictor('heart_rate_student.pt')
```

## 前端集成

在 `templates/index.html` 中添加KD结果显示：

```html
<!-- KD心率显示 -->
<div class="kd-heart-rate">
    <h3>KD预测心率</h3>
    <div class="value" id="kdHeartRate">--</div>
    <div class="unit">BPM</div>
</div>

<!-- KD波形图 -->
<div class="chart-container">
    <canvas id="kdWaveformChart"></canvas>
</div>
```

```javascript
// 更新KD数据
function updateKDData(data) {
    document.getElementById('kdHeartRate').textContent = 
        data.kd_hr.toFixed(1);
    
    // 更新KD波形
    if (kdWaveformChart) {
        kdWaveformChart.data.datasets[0].data = data.kd_waveform;
        kdWaveformChart.update();
    }
}
```

## 验证数据要求

### 训练数据格式
```
X_ecg: (batch, sequence_length, ecg_features=128)
X_radar: (batch, sequence_length, radar_features=64)  
y_hr: (batch,)  # 心率标签 (BPM)
```

### 序列长度建议
- ECG序列长度: 100-200个时间步
- Radar序列长度: 100个时间步
- 采样率: 20Hz (50ms间隔)

### 特征维度
- ECG特征: 128维 (相位、幅度、统计特征等)
- Radar特征: 64维 (相位、幅度、统计特征等)

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| MAE | < 5 BPM | 平均绝对误差 |
| RMSE | < 7 BPM | 均方根误差 |
| Inference Time | < 10 ms | 单帧推理时间 |

## 故障排除

### 问题1：内存不足
```python
# 减小批次大小
trainer.train(batch_size=16)
```

### 问题2：训练不稳定
```python
# 减小学习率
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

### 问题3：过拟合
```python
# 添加正则化
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 或使用Dropout
nn.Dropout(0.2)
```

### 问题4：模型太大
```python
# 减小模型尺寸
teacher_hidden_dim = 128
student_hidden_dim = 64
```

## 高级功能

### 1. 注意力可视化
```python
# 可视化教师和学生的注意力图
teacher_attn = teacher.get_attention_maps(ecg_sample)
student_attn = student.get_attention_maps(radar_sample)
```

### 2. 渐进式蒸馏
```python
# 逐步增加蒸馏强度
for epoch in range(num_epochs):
    if epoch < 10:
        lambda_feature = 0.1
    elif epoch < 20:
        lambda_feature = 0.3
    else:
        lambda_feature = 0.5
```

### 3. 多任务学习
```python
# 同时预测心率和呼吸率
L_total = L_hr + L_resp + 0.5*L_feature + 0.2*L_attention
```

## 参考文献

1. 原论文方法: "Cross-modal Knowledge Distillation for Radar-based Heart Rate Monitoring"
2. Transformer架构: "Attention Is All You Need" (Vaswani et al., 2017)
3. 知识蒸馏: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

## 联系方式

如有问题，请查看代码中的注释或创建Issue。

---
*更新日期: 2024年*
*版本: 1.0.0*
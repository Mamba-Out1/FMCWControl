"""
将跨模态蒸馏模型集成到心率监控系统
"""

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from typing import Optional, Tuple
import time


class KDHeartRatePredictor:
    """基于知识蒸馏的心率预测器"""
    
    def __init__(self, model_path: str = 'heart_rate_student.onnx'):
        """
        初始化预测器
        
        Args:
            model_path: ONNX模型路径，可选格式:
                - 'heart_rate_student.onnx' (推荐)
                - 'heart_rate_student.npz' (NumPy权重)
                - 'heart_rate_student.pt' (PyTorch权重)
        """
        self.model_path = model_path
        self.session = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输入输出配置
        self.input_seq_len = 100  # 序列长度
        self.input_feature_dim = 64  # 特征维度
        
        # 缓冲区
        self.feature_buffer = []
        self.buffer_size = 200  # 保持足够长的历史用于滑动窗口
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        if self.model_path.endswith('.onnx'):
            self._load_onnx_model()
        elif self.model_path.endswith('.npz'):
            self._load_npz_model()
        elif self.model_path.endswith('.pt'):
            self._load_pytorch_model()
        else:
            raise ValueError(f"不支持的文件格式: {self.model_path}")
        
        print(f"知识蒸馏模型已加载: {self.model_path}")
        print(f"输入维度: (batch, {self.input_seq_len}, {self.input_feature_dim})")
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        try:
            # 配置ONNX Runtime
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # 获取输入输出信息
            input_info = self.session.get_inputs()[0]
            self.input_seq_len = input_info.shape[1]
            self.input_feature_dim = input_info.shape[2]
            
            print(f"ONNX模型加载成功")
            print(f"输入形状: {input_info.shape}")
            print(f"输出: {[out.name for out in self.session.get_outputs()]}")
            
        except Exception as e:
            print(f"ONNX模型加载失败: {e}")
            print("尝试加载NumPy权重...")
            self._load_npz_fallback()
    
    def _load_npz_model(self):
        """从NumPy权重文件加载模型"""
        try:
            # 加载权重
            weights = np.load(self.model_path, allow_pickle=True)
            
            # 重建简化模型
            class SimpleKDModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_proj = nn.Linear(64, 128)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=128, nhead=4, 
                                                   dim_feedforward=256, dropout=0.1, 
                                                   batch_first=True),
                        num_layers=2
                    )
                    self.regressor = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    x_proj = self.input_proj(x)
                    
                    # 简单位置编码
                    seq_len = x_proj.size(1)
                    position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
                    x_proj = x_proj + (position / seq_len)
                    
                    encoded = self.transformer(x_proj)
                    pooled = encoded.mean(dim=1)
                    heart_rate = self.regressor(pooled)
                    return heart_rate
            
            self.model = SimpleKDModel().to(self.device)
            
            # 加载权重
            state_dict = {}
            for key, value in weights.items():
                state_dict[key] = torch.from_numpy(value)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print("NumPy权重加载成功")
            
        except Exception as e:
            print(f"NumPy模型加载失败: {e}")
            self._create_fallback_model()
    
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        try:
            # 定义模型结构
            class RadarTransformer(nn.Module):
                def __init__(self, input_dim=64, hidden_dim=128):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                                                   dim_feedforward=256, dropout=0.1,
                                                   batch_first=True),
                        num_layers=2
                    )
                    self.regressor = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    x_proj = self.input_proj(x)
                    seq_len = x_proj.size(1)
                    position = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
                    x_proj = x_proj + (position / seq_len)
                    
                    encoded = self.transformer(x_proj)
                    pooled = encoded.mean(dim=1)
                    heart_rate = self.regressor(pooled)
                    return heart_rate
            
            self.model = RadarTransformer().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            print("PyTorch模型加载成功")
            
        except Exception as e:
            print(f"PyTorch模型加载失败: {e}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """创建备用模型"""
        print("创建备用简化模型...")
        
        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU()
                )
                self.regressor = nn.Linear(16, 1)
            
            def forward(self, x):
                # 简化: 对序列维度取平均
                x_mean = x.mean(dim=1)
                features = self.encoder(x_mean)
                heart_rate = self.regressor(features)
                return heart_rate
        
        self.model = FallbackModel().to(self.device)
        self.model.eval()
        print("备用模型已创建")
    
    def _load_npz_fallback(self):
        """ONNX加载失败时的备用方案"""
        # 尝试加载NumPy版本
        npz_path = self.model_path.replace('.onnx', '.npz')
        if npz_path != self.model_path and not npz_path.endswith('.npz'):
            npz_path = 'heart_rate_student.npz'
        
        print(f"尝试加载备用文件: {npz_path}")
        self.model_path = npz_path
        self._load_npz_model()
    
    def extract_features(self, radar_data: np.ndarray) -> np.ndarray:
        """
        从雷达数据中提取特征
        
        Args:
            radar_data: 原始雷达数据，形状为 (chirps, samples) 或类似
            
        Returns:
            特征向量，长度为 self.input_feature_dim
        """
        # 简化特征提取: 相位信息、幅度、统计特征等
        # 在实际应用中，这应该更复杂，但这里保持简单
        
        if len(radar_data.shape) == 2:
            # 假设是 (chirps, samples)
            chirps, samples = radar_data.shape
            
            # 提取相位信息
            phase_data = np.angle(radar_data)
            
            # 计算统计特征
            features = []
            
            # 相位均值、方差
            phase_mean = np.mean(phase_data, axis=1)
            phase_std = np.std(phase_data, axis=1)
            
            # 幅度特征
            magnitude = np.abs(radar_data)
            mag_mean = np.mean(magnitude, axis=1)
            mag_std = np.std(magnitude, axis=1)
            
            # 组合特征
            features.extend(phase_mean)
            features.extend(phase_std)
            features.extend(mag_mean)
            features.extend(mag_std)
            
            # 如果特征不足，添加零填充
            if len(features) < self.input_feature_dim:
                features.extend([0] * (self.input_feature_dim - len(features)))
            elif len(features) > self.input_feature_dim:
                features = features[:self.input_feature_dim]
            
            return np.array(features, dtype=np.float32)
        
        else:
            # 如果数据形状不符，返回零向量
            print(f"警告: 意外的数据形状 {radar_data.shape}")
            return np.zeros(self.input_feature_dim, dtype=np.float32)
    
    def update_buffer(self, features: np.ndarray):
        """更新特征缓冲区"""
        self.feature_buffer.append(features)
        
        # 保持缓冲区大小
        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer = self.feature_buffer[-self.buffer_size:]
    
    def prepare_input(self) -> Optional[np.ndarray]:
        """准备模型输入"""
        if len(self.feature_buffer) < self.input_seq_len:
            return None
        
        # 取最新的序列
        sequence = np.array(self.feature_buffer[-self.input_seq_len:], dtype=np.float32)
        
        # 形状: (1, seq_len, feature_dim)
        sequence = sequence.reshape(1, self.input_seq_len, self.input_feature_dim)
        
        return sequence
    
    def predict_heart_rate(self, radar_data: np.ndarray) -> Optional[float]:
        """
        预测心率
        
        Args:
            radar_data: 当前雷达帧数据
            
        Returns:
            预测的心率值 (BPM)，如果数据不足则返回None
        """
        try:
            # 提取特征
            features = self.extract_features(radar_data)
            
            # 更新缓冲区
            self.update_buffer(features)
            
            # 准备输入
            model_input = self.prepare_input()
            if model_input is None:
                return None
            
            # 推理
            if self.session is not None:
                # ONNX推理
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                
                heart_rate = self.session.run([output_name], {input_name: model_input})[0]
                heart_rate = float(heart_rate[0, 0])
                
            elif self.model is not None:
                # PyTorch推理
                with torch.no_grad():
                    input_tensor = torch.from_numpy(model_input).to(self.device)
                    heart_rate_tensor = self.model(input_tensor)
                    heart_rate = float(heart_rate_tensor.cpu().numpy()[0, 0])
            else:
                # 没有可用模型
                return None
            
            # 限制心率范围 (30-200 BPM)
            heart_rate = max(30, min(200, heart_rate))
            
            return heart_rate
            
        except Exception as e:
            print(f"心率预测错误: {e}")
            return None
    
    def get_waveform(self, length: int = 100) -> np.ndarray:
        """获取最近的心率波形"""
        if not self.feature_buffer:
            return np.zeros(length)
        
        # 从特征缓冲区提取第一个特征维度作为波形
        waveform = [features[0] if len(features) > 0 else 0 
                   for features in self.feature_buffer[-length:]]
        
        # 标准化波形
        if len(waveform) > 0:
            waveform = np.array(waveform)
            if np.std(waveform) > 0:
                waveform = (waveform - np.mean(waveform)) / np.std(waveform)
        
        return waveform


def test_predictor():
    """测试预测器"""
    print("测试KDHeartRatePredictor...")
    
    # 创建预测器
    predictor = KDHeartRatePredictor()
    
    # 生成模拟雷达数据
    np.random.seed(42)
    test_data = np.random.randn(16, 128) + 1j * np.random.randn(16, 128)
    
    # 模拟连续预测
    for i in range(50):
        # 轻微修改数据以模拟变化
        test_data = test_data + 0.1 * np.random.randn(*test_data.shape)
        
        heart_rate = predictor.predict_heart_rate(test_data)
        
        if heart_rate is not None:
            print(f"帧 {i+1}: 预测心率 = {heart_rate:.1f} BPM")
        else:
            print(f"帧 {i+1}: 数据不足...")
        
        time.sleep(0.1)  # 模拟实时处理
    
    # 获取波形
    waveform = predictor.get_waveform()
    print(f"波形长度: {len(waveform)}, 均值: {np.mean(waveform):.2f}, 标准差: {np.std(waveform):.2f}")


def integrate_with_monitor():
    """与现有监控系统集成示例"""
    print("""
在heart_rate_monitor.py中集成示例:

1. 在RadarDataProcessor类的__init__方法中添加:
   self.kd_predictor = KDHeartRatePredictor('heart_rate_student.onnx')
   self.current_kd_hr = 0.0
   self.kd_waveform_buffer = deque(maxlen=200)

2. 在process_radar_data方法中添加KD推理:
   if self.kd_predictor:
       # 提取当前帧数据 (假设frame_data是复数雷达数据)
       kd_hr = self.kd_predictor.predict_heart_rate(frame_data)
       
       if kd_hr is not None:
           self.current_kd_hr = kd_hr
           
           # 更新波形
           kd_wave = self.kd_predictor.get_waveform(100)
           self.kd_waveform_buffer.extend(kd_wave)

3. 在get_data方法中返回KD结果:
   'kd_hr': float(self.current_kd_hr),
   'kd_waveform': list(self.kd_waveform_buffer)[-100:],

4. 在前端index.html中显示KD结果:
   - 添加KD心率显��
   - 添加KD波形图
    """)


if __name__ == "__main__":
    # 运行测试
    test_predictor()
    print("\n" + "=" * 60)
    integrate_with_monitor()
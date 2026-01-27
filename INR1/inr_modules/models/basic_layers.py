"""
基础网络组件
包含傅里叶特征编码等基础模块
"""
import torch
import torch.nn as nn
import numpy as np


class FourierFeatureEncoding(nn.Module):
    """
    傅里叶特征编码
    将低维坐标映射到高频波形空间
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions"
    """
    
    def __init__(self, input_dim, mapping_size=256, scale=1.0):
        """
        Args:
            input_dim: 输入维度
            mapping_size: 映射维度
            scale: 频率尺度
        """
        super().__init__()
        self.mapping_size = mapping_size
        
        # 固定随机矩阵
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    
    def forward(self, x):
        """
        Args:
            x: [Batch, input_dim]
            
        Returns:
            [Batch, 2*mapping_size] sin/cos特征
        """
        proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

"""
空间与时间基函数网络
"""
import torch
import torch.nn as nn
from .basic_layers import FourierFeatureEncoding


class SpatialBasisNet(nn.Module):
    """
    空间形态基函数网络
    
    输入: Lat, Lon, Alt, LocalTime特征
    输出: 原始空间基函数 Phi_raw(x)
    
    物理含义: 捕捉电离层垂直剖面形状和基本水平结构
    """
    
    def __init__(self, basis_dim=64, hidden_dim=256):
        """
        Args:
            basis_dim: 基函数维度
            hidden_dim: 隐层维度
        """
        super().__init__()
        
        # Input: Lat, Lon, Alt, SinLT, CosLT (5维)
        self.encoder = FourierFeatureEncoding(input_dim=5, mapping_size=128, scale=0.25)
        
        self.net = nn.Sequential(
            nn.Linear(256, hidden_dim), 
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, basis_dim)
        )
        
        # LayerNorm保证基函数分布稳定
        self.norm = nn.LayerNorm(basis_dim)
    
    def forward(self, coords_spatial):
        """
        Args:
            coords_spatial: [Batch, 5] (Lat, Lon, Alt, SinLT, CosLT)
            
        Returns:
            [Batch, basis_dim] 空间基函数
        """
        features = self.encoder(coords_spatial)
        basis = self.net(features)
        return self.norm(basis)


class TemporalBaseNet(nn.Module):
    """
    基准时间演化网络
    
    输入: UT Time
    输出: 基准时间系数 w_base(t)
    
    物理含义: 拟合太阳天顶角变化引起的气候态日变化
    """
    
    def __init__(self, basis_dim=64, hidden_dim=128):
        """
        Args:
            basis_dim: 基函数维度
            hidden_dim: 隐层维度
        """
        super().__init__()
        
        # Input: Normalized Time (1维)
        self.encoder = FourierFeatureEncoding(input_dim=1, mapping_size=64, scale=2.0)
        
        self.net = nn.Sequential(
            nn.Linear(128, hidden_dim), 
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, basis_dim)
        )
    
    def forward(self, time_norm):
        """
        Args:
            time_norm: [Batch, 1] 归一化时间
            
        Returns:
            [Batch, basis_dim] 时间基函数
        """
        features = self.encoder(time_norm)
        return self.net(features)

"""
物理引导的隐式神经辐射场 (Physics-Guided INR)
主模型架构
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .basis_networks import SpatialBasisNet, TemporalBaseNet
from .modulation_networks import SWPerturbationNet, VTECSpatialModulator


class PhysicsGuidedINR(nn.Module):
    """
    物理引导INR模型
        Ne = IRI + Alpha * Correction   
    其中:
        Correction = sum(w_total * basis_modulated)
        Alpha = f(Time, Kp, F10.7, TEC) 动态融合门控
    """
    
    def __init__(self, iri_proxy, lat_range, lon_range, alt_range,
                 sw_manager=None, tec_manager=None, start_date_str=None, config=None):
        """
        Args:
            iri_proxy: IRI神经代理场
            lat_range: 纬度范围
            lon_range: 经度范围
            alt_range: 高度范围
            sw_manager: 空间天气管理器
            tec_manager: TEC管理器
            start_date_str: 起始日期字符串
            config: 配置字典
        """
        super().__init__()
        
        # 物理参数
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.alt_range = alt_range
        self.total_hours = config.get('total_hours', 720.0)
        self.seq_len = config['seq_len']
        
        # 数据管理器（用于推理）
        self.sw_manager = sw_manager
        self.tec_manager = tec_manager
        self.start_date = pd.to_datetime(start_date_str) if start_date_str else None
        
        # IRI神经代理场（冻结）
        self.iri_proxy = iri_proxy
        for param in self.iri_proxy.parameters():
            param.requires_grad = False
        self.iri_proxy.eval()
        
        # 模型维度
        self.basis_dim = config.get('basis_dim', 64)
        
        # 空间基函数网络
        self.spatial_net = SpatialBasisNet(basis_dim=self.basis_dim)
        
        # 时间基网络
        self.temporal_net = TemporalBaseNet(basis_dim=self.basis_dim)
        
        # 空间天气弱调制
        self.sw_driver = SWPerturbationNet(
            basis_dim=self.basis_dim,
            d_model=config['d_model'],
            seq_len=self.seq_len
        )
        
        # VTEC空间调制
        self.vtec_modulator = VTECSpatialModulator(
            basis_dim=self.basis_dim,
            d_model=config['d_model'],
            seq_len=self.seq_len
        )
        
        # 动态融合门控网络 (Alpha Gate)
        self.alpha_gate_net = nn.Sequential(
            nn.Linear(self.basis_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 确保alpha在[0, 1]之间
        )
        
        # 不确定性投影层
        self.uncertainty_proj = nn.Linear(self.basis_dim, self.basis_dim)
        
        # 残差缩放参数
        self.resid_scale = nn.Parameter(torch.tensor(1.0))
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        # 时间网络零初始化
        nn.init.zeros_(self.temporal_net.net[-1].weight)
        nn.init.zeros_(self.temporal_net.net[-1].bias)
        
        # 不确定性投影
        nn.init.normal_(self.uncertainty_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.uncertainty_proj.bias)
        
        # Alpha gate
        nn.init.xavier_uniform_(self.alpha_gate_net[0].weight)
    
    def normalize_coords_internal(self, lat, lon, alt, time):
        """INR内部坐标归一化"""
        lat_n = lat / 90.0
        lon_n = lon / 180.0
        
        alt_min, alt_max = self.alt_range
        alt_n = 2.0 * (alt - alt_min) / (alt_max - alt_min) - 1.0
        
        time_n = (time / self.total_hours) * 2.0 - 1.0
        
        return lat_n, lon_n, alt_n, time_n
    
    def get_background(self, lat, lon, alt, time):
        """
        查询IRI Neural Proxy背景值
    
        关键修改：
        1. IRI Proxy参数frozen，但需要对输入coords保留梯度
        2. 临时切换到train模式以启用梯度传播
        """
        coords = torch.stack([lat, lon, alt, time], dim=-1)
    
        # ✅ 关键修复：临时启用train模式让梯度通过
        was_training = self.iri_proxy.training
        self.iri_proxy.train()  # 临时设为train模式
    
        try:
            background_log = self.iri_proxy(coords)
        finally:
            # 恢复原始状态
            if not was_training:
                self.iri_proxy.eval()
    
        return background_log
    
    def create_temporal_mask(self, current_time_batch):
        """生成时序掩码 (t < 0 为无效)"""
        device = current_time_batch.device
        offsets = torch.arange(self.seq_len - 1, -1, -1, device=device)
        history_times = current_time_batch.unsqueeze(1) - offsets.unsqueeze(0)
        return history_times < 0
    
    def forward(self, coords, sw_seq, vtec_seq):
        """
        前向传播
        
        Args:
            coords: [Batch, 4] (Lat, Lon, Alt, Time)
            sw_seq: [Batch, Seq, 2] 空间天气序列
            vtec_seq: [Batch, Seq, 1] VTEC序列
            
        Returns:
            output: [Batch, 1] 预测Ne (均值)
            log_var: [Batch, 1] 不确定性 (对数方差)
            final_correction: [Batch, 1] 最终残差
            delta_w: [Batch, basis_dim] SW扰动
            vtec_gamma: [Batch, basis_dim] VTEC尺度
            vtec_beta: [Batch, basis_dim] VTEC偏移
        """
        lat, lon, alt, time = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        
        # 1. 获取IRI背景
        background = self.get_background(lat, lon, alt, time)
        
        # 2. 特征准备
        lat_n, lon_n, alt_n, time_n = self.normalize_coords_internal(lat, lon, alt, time)
        
        # 地方时特征
        local_time_hour = (time % 24.0) + (lon / 15.0)
        lt_norm = local_time_hour / 24.0
        sin_lt = torch.sin(2 * np.pi * lt_norm)
        cos_lt = torch.cos(2 * np.pi * lt_norm)
        
        spatial_input = torch.stack([lat_n, lon_n, alt_n, sin_lt, cos_lt], dim=1)
        temporal_input = time_n.unsqueeze(1)
        
        # 时序掩码
        temporal_mask = self.create_temporal_mask(time)
        
        # --- A. 空间路径（受VTEC调制） ---
        basis_raw = self.spatial_net(spatial_input)
        vtec_gamma, vtec_beta = self.vtec_modulator(vtec_seq, mask=temporal_mask)
        basis_mod = basis_raw * (1.0 + vtec_gamma) + vtec_beta
        
        # --- B. 时间路径（受SW弱调制） ---
        w_base = self.temporal_net(temporal_input)
        delta_w = self.sw_driver(sw_seq, mask=temporal_mask)
        w_total = w_base + delta_w
        
        # --- C. 原始残差 ---
        raw_correction = torch.sum(w_total * basis_mod, dim=1, keepdim=True)
        raw_correction = torch.tanh(self.resid_scale * raw_correction)
        
        # --- D. 动态融合门控 Alpha ---
        gate_input = torch.cat([w_total, vtec_gamma], dim=-1)
        alpha = self.alpha_gate_net(gate_input)  # [Batch, 1]
        
        final_correction = alpha * raw_correction
        output = background + final_correction
        
        # --- E. 不确定性估计 ---
        w_var = self.uncertainty_proj(w_total)
        raw_log_var = torch.sum(w_var * basis_mod, dim=1, keepdim=True)
        log_var = torch.clamp(raw_log_var, -10.0, 10.0)
        
        return output, log_var, final_correction, delta_w, vtec_gamma, vtec_beta

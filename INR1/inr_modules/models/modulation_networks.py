"""
空间天气和VTEC调制网络
使用Transformer处理时序特征
"""
import torch
import torch.nn as nn


class SWPerturbationNet(nn.Module):
    """
    空间天气弱调制网络 (Kp/F10.7)
    
    输入: SW History [Batch, Seq, 2]
    输出: 时间系数扰动 delta_w(t)
    
    策略: 弱调制 (Tanh * scale)，仅修正演化系数，不改变空间结构
    """
    
    def __init__(self, basis_dim=64, d_model=64, nhead=4, num_layers=2, seq_len=12):
        """
        Args:
            basis_dim: 基函数维度
            d_model: Transformer隐层维度
            nhead: 注意力头数
            num_layers: Transformer层数
            seq_len: 序列长度
        """
        super().__init__()
        
        # Input: Kp, F10.7 (2维)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, basis_dim)
        )
        
        # 零初始化
        nn.init.zeros_(self.output_head[1].weight)
        nn.init.zeros_(self.output_head[1].bias)
    
    def forward(self, sw_seq, mask=None):
        """
        Args:
            sw_seq: [Batch, Seq, 2] 空间天气序列
            mask: [Batch, Seq] True表示padding
            
        Returns:
            [Batch, basis_dim] 扰动系数
        """
        # Mask处理：防止无效时序数据污染
        if mask is not None:
            sw_seq = sw_seq.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Transformer编码
        x = self.input_proj(sw_seq)
        x = x + self.pos_embed
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 取当前时刻上下文
        context_vector = x[:, -1, :]
        
        # 生成扰动系数 + 弱调制约束
        delta_w = self.output_head(context_vector)
        delta_w = torch.tanh(delta_w) * 0.5  # 限制幅度
        
        return delta_w


class VTECSpatialModulator(nn.Module):
    """
    VTEC全局空间调制网络
    
    输入: VTEC History [Batch, Seq, 1]
    输出: 空间调制参数 Gamma, Beta (FiLM风格)
    
    作用: 将VTEC数据映射为空间基函数的"增益"和"偏置"，实现水平梯度引导
    """
    
    def __init__(self, basis_dim=64, d_model=64, nhead=4, num_layers=2, seq_len=12):
        """
        Args:
            basis_dim: 基函数维度
            d_model: Transformer隐层维度
            nhead: 注意力头数
            num_layers: Transformer层数
            seq_len: 序列长度
        """
        super().__init__()
        
        # Input: VTEC (1维)
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # FiLM生成头: 生成Gamma (Scale) 和 Beta (Shift)
        self.film_generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, basis_dim * 2)
        )
        
        # 初始化：Gamma初始化为0 (对应Scale=1), Beta初始化为0
        nn.init.zeros_(self.film_generator[1].weight)
        nn.init.zeros_(self.film_generator[1].bias)
    
    def forward(self, vtec_seq, mask=None):
        """
        Args:
            vtec_seq: [Batch, Seq, 1] VTEC序列
            mask: [Batch, Seq] True表示padding
            
        Returns:
            gamma: [Batch, basis_dim] 尺度调制
            beta: [Batch, basis_dim] 偏移调制
        """
        # Mask保护
        if mask is not None:
            vtec_seq = vtec_seq.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Transformer编码
        x = self.input_proj(vtec_seq)
        x = x + self.pos_embed
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 取当前时刻上下文
        context_vector = x[:, -1, :]
        
        # 生成调制参数
        film_params = self.film_generator(context_vector)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # 约束Gamma防止过度缩放 (Scale range approx 0.8 ~ 1.2)
        gamma = torch.tanh(gamma) * 0.2
        
        return gamma, beta

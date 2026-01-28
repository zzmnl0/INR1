"""
循环神经网络组件

包含：
    1. GlobalEnvEncoder: LSTM 编码器，处理 Kp/F10.7 全局环境序列
    2. SpatialContextEncoder: ConvLSTM 编码器，处理 TEC 地图序列
"""

import torch
import torch.nn as nn


# ======================== ConvLSTM Cell ========================
class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元（单时间步）

    将标准 LSTM 的全连接操作替换为卷积操作，用于处理空间数据

    Reference:
        Shi et al., "Convolutional LSTM Network: A Machine Learning Approach
        for Precipitation Nowcasting" NIPS 2015
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Args:
            input_dim: 输入通道数
            hidden_dim: 隐层通道数
            kernel_size: 卷积核大小
            bias: 是否使用偏置
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # ConvLSTM 的 4 个门：输入门、遗忘门、单元门、输出门
        # 使用单个卷积层一次性计算所有门（效率优化）
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # i, f, g, o
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        单时间步前向传播

        Args:
            input_tensor: [Batch, Channels, H, W] 当前时间步输入
            cur_state: (h_cur, c_cur) 当前隐状态和单元状态
                h_cur: [Batch, Hidden_Dim, H, W]
                c_cur: [Batch, Hidden_Dim, H, W]

        Returns:
            h_next: [Batch, Hidden_Dim, H, W] 下一时刻隐状态
            c_next: [Batch, Hidden_Dim, H, W] 下一时刻单元状态
        """
        h_cur, c_cur = cur_state

        # 拼接输入和隐状态
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 计算所有门的值
        combined_conv = self.conv(combined)

        # 分离 4 个门
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 门控计算
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        g = torch.tanh(cc_g)     # 候选单元状态
        o = torch.sigmoid(cc_o)  # 输出门

        # 更新单元状态
        c_next = f * c_cur + i * g

        # 更新隐状态
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        """
        初始化隐状态和单元状态

        Args:
            batch_size: 批次大小
            image_size: (H, W)
            device: torch.device

        Returns:
            (h0, c0): 初始隐状态和单元状态
        """
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


# ======================== ConvLSTM 序列编码器 ========================
class ConvLSTM(nn.Module):
    """
    ConvLSTM 序列编码器（多时间步）

    处理完整序列并返回最后一帧的特征图
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        """
        Args:
            input_dim: 输入通道数
            hidden_dim: 隐层通道数（可以是 list，每层不同）
            kernel_size: 卷积核大小
            num_layers: ConvLSTM 层数
            bias: 是否使用偏置
        """
        super().__init__()

        # 参数检查
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers

        assert len(hidden_dim) == num_layers, "hidden_dim 长度必须等于 num_layers"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        # 构建 ConvLSTM 层
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size,
                    bias=self.bias
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        前向传播处理完整序列

        Args:
            input_tensor: [Batch, Seq_Len, Channels, H, W]
            hidden_state: 初始隐状态（可选）

        Returns:
            layer_output: [Batch, Hidden_Dim[-1], H, W] 最后一帧的最后一层特征
            last_state_list: List[(h, c)] 每层的最终状态
        """
        b, seq_len, _, h, w = input_tensor.size()

        # 初始化隐状态
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w),
                                              device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        # 逐层处理
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            # 逐时间步处理
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=(h, c)
                )
                output_inner.append(h)

            # 将时间步输出堆叠为新的序列
            layer_output = torch.stack(output_inner, dim=1)  # [B, T, C, H, W]
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # 返回最后一层的最后一帧特征
        return layer_output_list[-1][:, -1, :, :, :], last_state_list

    def _init_hidden(self, batch_size, image_size, device):
        """初始化所有层的隐状态"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size, device)
            )
        return init_states


# ======================== 全局环境编码器 (LSTM) ========================
class GlobalEnvEncoder(nn.Module):
    """
    全局环境编码器（处理 Kp/F10.7）

    输入: [Batch, Seq_Len, 2] (Kp, F10.7)
    输出: [Batch, Hidden_Dim] 全局环境上下文向量

    作用:
        - 提取磁暴和太阳活动的时序演变特征
        - 用于时间基函数的加性调制（Shift）
    """

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: 输入维度（Kp + F10.7 = 2）
            hidden_dim: LSTM 隐层维度
            num_layers: LSTM 层数
            dropout: Dropout 概率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # 输出投影（取最后时刻的隐状态）
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, sw_seq, mask=None):
        """
        前向传播

        Args:
            sw_seq: [Batch, Seq_Len, 2] Kp/F10.7 序列
            mask: [Batch, Seq_Len] 可选的时序掩码（True 表示无效）

        Returns:
            z_env: [Batch, Hidden_Dim] 全局环境上下文向量
        """
        # 如果有掩码，将无效位置置零
        if mask is not None:
            sw_seq = sw_seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # LSTM 编码
        # output: [Batch, Seq, Hidden_Dim]
        # h_n: [Num_Layers, Batch, Hidden_Dim]
        output, (h_n, c_n) = self.lstm(sw_seq)

        # 取最后一层的最后时刻隐状态
        z_env = h_n[-1]  # [Batch, Hidden_Dim]

        # 投影
        z_env = self.output_proj(z_env)

        return z_env


# ======================== 空间上下文编码器 (ConvLSTM) ========================
class SpatialContextEncoder(nn.Module):
    """
    空间上下文编码器（处理 TEC 地图序列）

    输入: [Batch, Seq_Len, 1, H, W] TEC Map 序列
    输出: [Batch, Feat_Dim, H, W] 空间特征图

    作用:
        - 提取 TEC 水平梯度的时空演变模式
        - 用于空间基函数的乘性调制（Scale）
    """

    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, kernel_size=3):
        """
        Args:
            input_dim: 输入通道数（TEC = 1）
            hidden_dim: ConvLSTM 隐层通道数
            num_layers: ConvLSTM 层数
            kernel_size: 卷积核大小
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # ConvLSTM 编码器
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=True
        )

        # 特征细化（可选）
        self.feature_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

    def forward(self, tec_map_seq):
        """
        前向传播

        Args:
            tec_map_seq: [Batch, Seq_Len, 1, H, W] TEC 地图序列

        Returns:
            F_tec: [Batch, Hidden_Dim, H, W] 空间特征图
        """
        # ConvLSTM 编码（返回最后一帧）
        F_tec, _ = self.convlstm(tec_map_seq)

        # 特征细化
        F_tec = self.feature_refine(F_tec)

        return F_tec


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("="*60)
    print("循环模块测试")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 GlobalEnvEncoder
    print("\n[测试 1] GlobalEnvEncoder (LSTM)")
    env_encoder = GlobalEnvEncoder(input_dim=2, hidden_dim=64, num_layers=2).to(device)
    sw_seq = torch.randn(16, 6, 2).to(device)  # [Batch=16, Seq=6, Feat=2]
    z_env = env_encoder(sw_seq)
    print(f"  输入: {sw_seq.shape}")
    print(f"  输出: {z_env.shape}")
    print(f"  参数量: {sum(p.numel() for p in env_encoder.parameters()):,}")

    # 测试 SpatialContextEncoder
    print("\n[测试 2] SpatialContextEncoder (ConvLSTM)")
    spatial_encoder = SpatialContextEncoder(
        input_dim=1, hidden_dim=32, num_layers=2, kernel_size=3
    ).to(device)
    tec_seq = torch.randn(8, 6, 1, 181, 361).to(device)  # [B=8, Seq=6, C=1, H=181, W=361]
    F_tec = spatial_encoder(tec_seq)
    print(f"  输入: {tec_seq.shape}")
    print(f"  输出: {F_tec.shape}")
    print(f"  参数量: {sum(p.numel() for p in spatial_encoder.parameters()):,}")

    # 测试梯度流动
    print("\n[测试 3] 梯度流动")
    loss = z_env.sum() + F_tec.sum()
    loss.backward()
    print(f"  LSTM 梯度范数: {env_encoder.lstm.weight_ih_l0.grad.norm().item():.6f}")
    print(f"  ConvLSTM 梯度范数: {spatial_encoder.convlstm.cell_list[0].conv.weight.grad.norm().item():.6f}")

    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)

"""
R-STMRF 滑动窗口数据整合工具

设计思路：
    - 保留现有的 FY_dataloader.py 时间分箱策略（TimeBinSampler）
    - 在训练循环中，通过 sw_manager 和 tec_manager 动态获取历史序列
    - 提供辅助函数简化数据流处理
"""

import torch
import numpy as np


class SlidingWindowBatchProcessor:
    """
    滑动窗口批次处理器

    职责：
        - 接收 FY 数据批次（原始格式）
        - 动态查询 sw_manager 和 tec_manager
        - 返回 R-STMRF 所需的完整数据包

    使用方式：
        在训练循环中，替代原有的数据预处理逻辑
    """

    def __init__(self, sw_manager, tec_manager, device='cuda'):
        """
        Args:
            sw_manager: SpaceWeatherManager 实例
            tec_manager: TECDataManager 实例
            device: 计算设备
        """
        self.sw_manager = sw_manager
        self.tec_manager = tec_manager
        self.device = device

    def process_batch(self, batch_data):
        """
        处理一个批次的 FY 数据（优化版：识别唯一时间窗口，避免重复计算 ConvLSTM）

        Args:
            batch_data: [Batch, 5] Tensor
                Columns: [Lat, Lon, Alt, Time, Ne_Log]

        Returns:
            coords: [Batch, 4] (Lat, Lon, Alt, Time)
            target_ne: [Batch, 1] 真值 Ne（对数）
            sw_seq: [Batch, Seq, 2] 空间天气序列
            unique_tec_map_seq: [N_unique, Seq, 1, H, W] 唯一时间窗口的 TEC 地图序列
            tec_indices: [Batch] 每个样本对应的唯一时间窗口索引
            target_tec_map: [Batch, 1, H, W] 当前时刻 TEC 地图
        """
        # 解析批次数据
        lat = batch_data[:, 0].to(self.device)
        lon = batch_data[:, 1].to(self.device)
        alt = batch_data[:, 2].to(self.device)
        time = batch_data[:, 3].to(self.device)
        target_ne = batch_data[:, 4].unsqueeze(1).to(self.device)

        # 构建坐标
        coords = torch.stack([lat, lon, alt, time], dim=1)  # [Batch, 4]

        # 查询空间天气序列（点级，不需要去重）
        sw_seq = self.sw_manager.get_drivers_sequence(time)  # [Batch, Seq, 2]

        # ==================== 关键优化：识别唯一时间窗口 ====================
        # TEC 是区域级背景场，应该被所有 query points 共享
        # 1. 识别 batch 内的唯一时间索引
        time_indices_int = time.round().long()  # 将时间四舍五入为整数索引
        unique_time_indices, inverse_indices = torch.unique(time_indices_int, return_inverse=True)
        # unique_time_indices: [N_unique] 唯一的时间索引
        # inverse_indices: [Batch] 每个样本对应的唯一索引位置

        # 2. 只对唯一时间窗口查询 TEC 地图序列
        # 创建唯一时间的张量用于查询
        unique_times = unique_time_indices.float()  # 转回浮点数
        unique_tec_map_seq = self.tec_manager.get_tec_map_sequence(unique_times)
        # unique_tec_map_seq: [N_unique, Seq, 1, H, W]

        # 3. 为每个样本提取对应的当前时刻 TEC 地图（用于 Loss 计算）
        # 使用 inverse_indices 索引对应的 TEC map
        target_tec_map = unique_tec_map_seq[inverse_indices, -1, :, :, :]  # [Batch, 1, H, W]

        return coords, target_ne, sw_seq, unique_tec_map_seq, inverse_indices, target_tec_map


def get_r_stmrf_dataloaders(fy_path, val_days, batch_size, bin_size_hours,
                              sw_manager, tec_manager, num_workers=0):
    """
    获取 R-STMRF 专用的 DataLoader

    Note:
        实际上使用原有的 get_dataloaders 即可，数据处理在训练循环中进行

    Args:
        fy_path: FY 数据路径
        val_days: 验证集日期
        batch_size: 批次大小
        bin_size_hours: 时间分箱大小
        sw_manager: 空间天气管理器
        tec_manager: TEC 管理器
        num_workers: 工作进程数

    Returns:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        batch_processor: 批次处理器实例
    """
    # 导入原有的 DataLoader 工厂函数
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from data_managers.FY_dataloader import get_dataloaders

    # 获取原有的 DataLoader
    train_loader, val_loader = get_dataloaders(
        npy_path=fy_path,
        val_days=val_days,
        batch_size=batch_size,
        bin_size_hours=bin_size_hours,
        num_workers=num_workers
    )

    # 创建批次处理器
    device = next(iter(train_loader)).device if torch.cuda.is_available() else 'cpu'
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    return train_loader, val_loader, batch_processor


# ======================== 辅助函数 ========================
def collate_with_sequences(batch, sw_manager, tec_manager, device='cuda'):
    """
    自定义 Collate 函数（可选方案）

    如果希望在 DataLoader 层面整合序列数据，可以使用此函数作为 collate_fn

    Args:
        batch: List of samples from Dataset
        sw_manager: SpaceWeatherManager
        tec_manager: TECDataManager
        device: 计算设备

    Returns:
        collated_data: dict
    """
    # 将批次堆叠为 Tensor
    batch_data = torch.stack([item for item in batch], dim=0)  # [Batch, 5]

    # 使用批次处理器
    processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)
    coords, target_ne, sw_seq, tec_map_seq, target_tec_map = processor.process_batch(batch_data)

    return {
        'coords': coords,
        'target_ne': target_ne,
        'sw_seq': sw_seq,
        'tec_map_seq': tec_map_seq,
        'target_tec_map': target_tec_map
    }


# ======================== 使用示例 ========================
if __name__ == '__main__':
    print("="*60)
    print("滑动窗口数据处理器测试")
    print("="*60)

    # 模拟数据管理器
    class DummySWManager:
        def __init__(self, seq_len=6, device='cuda'):
            self.seq_len = seq_len
            self.device = device

        def get_drivers_sequence(self, time_batch):
            batch_size = time_batch.shape[0]
            return torch.randn(batch_size, self.seq_len, 2).to(self.device)

    class DummyTECManager:
        def __init__(self, seq_len=6, device='cuda'):
            self.seq_len = seq_len
            self.device = device

        def get_tec_map_sequence(self, time_batch):
            batch_size = time_batch.shape[0]
            return torch.rand(batch_size, self.seq_len, 1, 181, 361).to(self.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建管理器
    sw_manager = DummySWManager(seq_len=6, device=device)
    tec_manager = DummyTECManager(seq_len=6, device=device)

    # 创建批次处理器
    processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device=device)

    # 模拟 FY 批次数据
    batch_size = 128
    batch_data = torch.randn(batch_size, 5)
    batch_data[:, 0] = batch_data[:, 0] * 90  # Lat
    batch_data[:, 1] = batch_data[:, 1] * 180  # Lon
    batch_data[:, 2] = 200 + batch_data[:, 2].abs() * 100  # Alt
    batch_data[:, 3] = batch_data[:, 3].abs() * 100  # Time
    batch_data[:, 4] = batch_data[:, 4] + 11.0  # Ne_Log

    print(f"\n输入批次数据: {batch_data.shape}")

    # 处理批次
    coords, target_ne, sw_seq, unique_tec_map_seq, tec_indices, target_tec_map = processor.process_batch(batch_data)

    print("\n输出形状:")
    print(f"  coords: {coords.shape}")
    print(f"  target_ne: {target_ne.shape}")
    print(f"  sw_seq: {sw_seq.shape}")
    print(f"  unique_tec_map_seq: {unique_tec_map_seq.shape}")
    print(f"  tec_indices: {tec_indices.shape}")
    print(f"  target_tec_map: {target_tec_map.shape}")

    print("\n唯一时间窗口统计:")
    print(f"  Batch size: {batch_size}")
    print(f"  唯一时间窗口数: {unique_tec_map_seq.shape[0]}")
    print(f"  重复率: {batch_size / unique_tec_map_seq.shape[0]:.2f}x")

    print("\n数据范围检查:")
    print(f"  Lat: [{coords[:, 0].min().item():.2f}, {coords[:, 0].max().item():.2f}]")
    print(f"  Lon: [{coords[:, 1].min().item():.2f}, {coords[:, 1].max().item():.2f}]")
    print(f"  Alt: [{coords[:, 2].min().item():.2f}, {coords[:, 2].max().item():.2f}]")
    print(f"  Time: [{coords[:, 3].min().item():.2f}, {coords[:, 3].max().item():.2f}]")

    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)

"""
R-STMRF 训练脚本

完整的训练流程，包括：
    - 数据加载（保留 TimeBinSampler）
    - 模型初始化
    - 物理约束损失
    - 训练和验证循环
    - 模型保存
"""

import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config_r_stmrf import get_config_r_stmrf, print_config_r_stmrf
from r_stmrf_model import R_STMRF_Model
from physics_losses_r_stmrf import combined_physics_loss
from sliding_dataset import SlidingWindowBatchProcessor

from data_managers import SpaceWeatherManager, TECDataManager, IRINeuralProxy
from data_managers.FY_dataloader import FY3D_Dataset, TimeBinSampler


class SubsetTimeBinSampler(TimeBinSampler):
    """
    TimeBinSampler 的 Subset 版本

    用于在 random_split 之后仍然使用时间分箱策略
    """
    def __init__(self, subset: Subset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        # 获取底层的 FY3D_Dataset
        base_dataset = subset.dataset
        subset_indices = subset.indices

        # 构建 Subset 的 indices_by_bin
        # 只保留在 subset 中的索引
        subset_indices_set = set(subset_indices)
        filtered_indices_by_bin = {}

        for bin_id, indices in base_dataset.indices_by_bin.items():
            # 过滤出在 subset 中的索引
            filtered_indices = np.array([idx for idx in indices if idx in subset_indices_set])
            if len(filtered_indices) > 0:
                filtered_indices_by_bin[bin_id] = filtered_indices

        # 创建一个临时的 dataset 对象，只用于存储 indices_by_bin
        class TempDataset:
            def __init__(self, indices_by_bin):
                self.indices_by_bin = indices_by_bin

        temp_dataset = TempDataset(filtered_indices_by_bin)

        # 调用父类初始化（但使用临时 dataset）
        self.dataset = temp_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


def train_one_epoch(model, train_loader, batch_processor, optimizer, device, config, epoch, use_tec_cache=False):
    """
    训练一个 epoch

    Args:
        model: R-STMRF 模型
        train_loader: 训练数据 loader
        batch_processor: 批次处理器
        optimizer: 优化器
        device: 设备
        config: 配置字典
        epoch: 当前 epoch
        use_tec_cache: 是否使用TEC缓存（多时间尺度优化）

    Returns:
        avg_loss: 平均损失
        loss_dict: 各项损失的字典
    """
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_physics = 0.0
    total_chapman = 0.0
    total_tec_direction = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, batch_data in enumerate(pbar):
        # 1. 处理批次数据（获取序列，识别唯一时间窗口）
        coords, target_ne, sw_seq, unique_tec_map_seq, tec_indices, target_tec_map = batch_processor.process_batch(batch_data)

        # 判断是否需要计算物理损失（间歇性计算以加速训练）
        physics_loss_freq = config.get('physics_loss_freq', 10)  # 默认每10个batch计算一次
        compute_physics = (batch_idx % physics_loss_freq == 0)

        # 只在需要物理损失时启用梯度
        if compute_physics:
            coords.requires_grad_(True)

        # 2. 前向传播（支持两种模式：缓存优化 / 向后兼容）
        if use_tec_cache:
            # 多时间尺度优化：使用缓存，ConvLSTM只在整点小时运行
            pred_ne, log_var, correction, extras = model(coords, sw_seq)
        else:
            # 向后兼容：直接处理唯一窗口（ConvLSTM不随batch增长）
            pred_ne, log_var, correction, extras = model(coords, sw_seq, unique_tec_map_seq, tec_indices)

        # 3. 计算主损失（MSE 或 Huber）
        if config['use_uncertainty']:
            # 异方差损失
            precision = torch.exp(-log_var)
            mse_term = (pred_ne - target_ne) ** 2
            loss_main = torch.mean(0.5 * precision * mse_term + 0.5 * log_var)
        else:
            # 简单 MSE
            loss_main = F.mse_loss(pred_ne, target_ne)

        # 4. 计算物理约束损失（间歇性计算）
        if compute_physics:
            loss_physics, physics_dict = combined_physics_loss(
                pred_ne=pred_ne,
                coords=coords,
                tec_grad_direction=extras.get('tec_grad_direction'),  # 新设计
                coords_normalized=extras.get('coords_normalized'),  # 新设计
                w_chapman=config['w_chapman'],
                w_tec_direction=config.get('w_tec_direction', 0.05),  # 新设计 - 梯度方向权重
                target_tec_map=target_tec_map,  # 兼容旧设计
                w_tec_align=config.get('w_tec_align', 0.0),  # 旧设计已弃用，设为 0
                tec_lat_range=config['lat_range'],
                tec_lon_range=config['lon_range']
            )
        else:
            # 跳过物理损失计算，使用零损失
            loss_physics = 0.0
            physics_dict = {
                'physics_total': 0.0,
                'chapman': 0.0,
                'tec_direction': 0.0
            }

        # 5. 总损失
        loss = config['w_mse'] * loss_main + loss_physics

        # 6. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # 7. 统计
        total_loss += loss.item()
        total_mse += loss_main.item()
        total_physics += physics_dict['physics_total']
        total_chapman += physics_dict['chapman']
        total_tec_direction += physics_dict.get('tec_direction', 0.0)
        num_batches += 1

        # 更新进度条
        physics_str = f"{physics_dict['physics_total']:.4f}" if compute_physics else "skip"
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'MSE': f"{loss_main.item():.4f}",
            'Physics': physics_str
        })

    # 平均损失
    avg_loss = total_loss / num_batches
    loss_dict = {
        'total': avg_loss,
        'mse': total_mse / num_batches,
        'physics': total_physics / num_batches,
        'chapman': total_chapman / num_batches,
        'tec_direction': total_tec_direction / num_batches
    }

    return avg_loss, loss_dict


def validate(model, val_loader, batch_processor, device, config, use_tec_cache=False):
    """
    验证模型

    Args:
        model: R-STMRF 模型
        val_loader: 验证数据 loader
        batch_processor: 批次处理器
        device: 设备
        config: 配置字典
        use_tec_cache: 是否使用TEC缓存（多时间尺度优化）

    Returns:
        avg_loss: 平均损失
        metrics: 评估指标字典
    """
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validating", leave=False):
            # 处理批次（识别唯一时间窗口）
            coords, target_ne, sw_seq, unique_tec_map_seq, tec_indices, target_tec_map = batch_processor.process_batch(batch_data)

            # 前向传播（支持两种模式）
            if use_tec_cache:
                pred_ne, log_var, correction, extras = model(coords, sw_seq)
            else:
                pred_ne, log_var, correction, extras = model(coords, sw_seq, unique_tec_map_seq, tec_indices)

            # MSE 损失
            loss_mse = F.mse_loss(pred_ne, target_ne)

            total_loss += loss_mse.item()
            total_mse += loss_mse.item()
            num_batches += 1

            # 收集预测和真值（用于计算指标）
            all_preds.append(pred_ne.cpu())
            all_targets.append(target_ne.cpu())

    # 计算平均损失
    avg_loss = total_loss / num_batches

    # 计算评估指标
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

    metrics = {
        'loss': avg_loss,
        'mse': total_mse / num_batches,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    return avg_loss, metrics


def train_r_stmrf(config):
    """
    主训练函数

    Args:
        config: 配置字典

    Returns:
        model: 训练好的模型
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        相关管理器
    """
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device(config['device'])
    print(f"\n{'='*70}")
    print(f"R-STMRF 训练流程")
    print(f"使用设备: {device}")
    print(f"{'='*70}\n")

    # ==================== 1. 初始化数据管理器 ====================
    print("[步骤 1] 初始化数据管理器...")

    sw_manager = SpaceWeatherManager(
        txt_path=config['sw_path'],
        start_date_str=config['start_date_str'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device
    )

    tec_manager = TECDataManager(
        tec_map_path=config['tec_path'],
        total_hours=config['total_hours'],
        seq_len=config['seq_len'],
        device=device
        # 保持原始分辨率73×73，不再使用降采样
    )

    # ==================== 2. 加载 IRI 神经代理 ====================
    print("\n[步骤 2] 加载 IRI 神经代理场...")

    if not os.path.exists(config['iri_proxy_path']):
        raise FileNotFoundError(f"IRI 代理未找到: {config['iri_proxy_path']}")

    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    state_dict = torch.load(config['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(state_dict)
    iri_proxy.eval()
    print("  ✓ IRI 代理已加载并冻结")

    # ==================== 3. 准备数据集（随机划分）====================
    print("\n[步骤 3] 准备数据集（随机划分）...")

    # 加载全部数据
    full_dataset = FY3D_Dataset(
        npy_path=config['fy_path'],
        mode='train',
        val_days=[],  # 不使用日期过滤，加载全部数据
        bin_size_hours=config['bin_size_hours']
    )

    # 随机划分
    total_samples = len(full_dataset)
    val_size = int(total_samples * config['val_ratio'])
    train_size = total_samples - val_size

    print(f"  总样本: {total_samples}")
    print(f"  训练集: {train_size} | 验证集: {val_size}")

    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # 创建时间分箱 Sampler（优化 ConvLSTM 内存使用）
    # 通过时间分箱，确保每个 batch 内的样本来自相似的时间窗口
    # 这样可以最大化唯一时间窗口的重复率，减少 ConvLSTM 的重复计算
    train_sampler = SubsetTimeBinSampler(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False
    )
    val_sampler = SubsetTimeBinSampler(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False
    )

    # 创建 DataLoader（使用 batch_sampler）
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"  训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}")
    print(f"  使用时间分箱策略优化 ConvLSTM 内存使用")

    # 创建批次处理器
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    # ==================== 4. 初始化模型 ====================
    print("\n[步骤 4] 初始化 R-STMRF 模型...")

    model = R_STMRF_Model(
        iri_proxy=iri_proxy,
        lat_range=config['lat_range'],
        lon_range=config['lon_range'],
        alt_range=config['alt_range'],
        sw_manager=sw_manager,
        tec_manager=tec_manager,
        start_date_str=config['start_date_str'],
        config=config,
        tec_cache=None  # 先不传缓存，等模型创建后再初始化
    ).to(device)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 4.5. 初始化小时级TEC缓存（可选优化）====================
    # 重要：必须在模型创建后初始化，以使用模型内部的 ConvLSTM 和梯度头
    use_tec_cache = config.get('use_tec_cache', False)  # 默认关闭，保持向后兼容

    if use_tec_cache:
        print("\n[步骤 4.5] 初始化小时级TEC缓存（多时间尺度优化）...")
        from .hourly_tec_cache import HourlyTECContextCache

        # 使用模型内部的 ConvLSTM 和梯度头（参数共享！）
        tec_cache = HourlyTECContextCache(
            convlstm_encoder=model.spatial_context_encoder,
            tec_gradient_head=model.tec_gradient_direction_head,
            max_cache_size=config.get('tec_cache_size', 100),
            device=device
        )

        # 将缓存绑定到模型
        model.tec_cache = tec_cache

        print(f"  ✓ TEC缓存初始化完成（最大缓存: {config.get('tec_cache_size', 100)} 小时）")
        print(f"  ✓ ConvLSTM 将只在整点小时运行，使用余弦平方插值到分钟级")
        print(f"  ✓ 缓存与模型参数共享，确保训练时一致性")
        print(f"  ✓ 训练模式：跳过缓存（保留梯度）| 评估模式：启用缓存（加速推理）")

    # ==================== 5. 优化器和调度器 ====================
    print("\n[步骤 5] 配置优化器...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    if config['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
    else:
        scheduler = None

    # ==================== 6. 训练循环 ====================
    print("\n[步骤 6] 开始训练...")
    physics_freq = config.get('physics_loss_freq', 10)
    if physics_freq > 1:
        print(f"  ⚡ 物理损失间歇性计算：每 {physics_freq} 个batch计算一次（加速训练）")
        print(f"  ⚡ 预期加速: ~{physics_freq/2:.1f}× 梯度计算减少")
    else:
        print(f"  📊 物理损失每个batch计算（physics_loss_freq=1）")
    print(f"{'='*70}\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*70}")

        # 训练
        train_loss, train_dict = train_one_epoch(
            model, train_loader, batch_processor, optimizer, device, config, epoch, use_tec_cache
        )
        train_losses.append(train_loss)

        # 验证
        val_loss, val_metrics = validate(model, val_loader, batch_processor, device, config, use_tec_cache)
        val_losses.append(val_loss)

        # 打印结果
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"    - MSE: {train_dict['mse']:.6f}")
        print(f"    - Physics: {train_dict['physics']:.6f}")
        print(f"      · Chapman: {train_dict['chapman']:.6f}")
        print(f"      · TEC Direction: {train_dict['tec_direction']:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"    - MAE: {val_metrics['mae']:.6f}")
        print(f"    - RMSE: {val_metrics['rmse']:.6f}")
        print(f"    - R²: {val_metrics['r2']:.4f}")

        # TEC缓存统计（如果启用）
        if use_tec_cache and tec_cache is not None:
            cache_stats = tec_cache.get_stats()
            print(f"  TEC缓存统计:")
            print(f"    - 命中率: {cache_stats['hit_rate']*100:.1f}% "
                  f"({cache_stats['hits']}/{cache_stats['total_queries']})")
            print(f"    - 缓存大小: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(config['save_dir'], 'best_r_stmrf_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 保存最佳模型: {save_path}")
        else:
            patience_counter += 1

        # 早停
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"\n早停触发！验证损失连续 {config['patience']} 轮未改善")
            break

        # 定期保存
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['save_dir'], f'r_stmrf_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"{'='*70}\n")

    return model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager


# ======================== 主函数 ========================
if __name__ == '__main__':
    # 获取配置
    config = get_config_r_stmrf()
    print_config_r_stmrf()

    # 开始训练
    model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager = train_r_stmrf(config)

    print("\n训练脚本执行完毕！")

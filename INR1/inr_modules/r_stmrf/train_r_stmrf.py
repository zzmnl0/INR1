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
from data_managers.FY_dataloader import get_dataloaders


def train_one_epoch(model, train_loader, batch_processor, optimizer, device, config, epoch):
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

    Returns:
        avg_loss: 平均损失
        loss_dict: 各项损失的字典
    """
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_physics = 0.0
    total_chapman = 0.0
    total_tec_align = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, batch_data in enumerate(pbar):
        # 1. 处理批次数据（获取序列）
        coords, target_ne, sw_seq, tec_map_seq, target_tec_map = batch_processor.process_batch(batch_data)

        # 确保 coords 需要梯度（用于物理损失）
        coords.requires_grad_(True)

        # 2. 前向传播
        pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_map_seq)

        # 3. 计算主损失（MSE 或 Huber）
        if config['use_uncertainty']:
            # 异方差损失
            precision = torch.exp(-log_var)
            mse_term = (pred_ne - target_ne) ** 2
            loss_main = torch.mean(0.5 * precision * mse_term + 0.5 * log_var)
        else:
            # 简单 MSE
            loss_main = F.mse_loss(pred_ne, target_ne)

        # 4. 计算物理约束损失
        loss_physics, physics_dict = combined_physics_loss(
            pred_ne=pred_ne,
            coords=coords,
            target_tec_map=target_tec_map,
            w_chapman=config['w_chapman'],
            w_tec_align=config['w_tec_align'],
            tec_lat_range=config['lat_range'],
            tec_lon_range=config['lon_range']
        )

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
        total_tec_align += physics_dict['tec_align']
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'MSE': f"{loss_main.item():.4f}",
            'Physics': f"{physics_dict['physics_total']:.4f}"
        })

    # 平均损失
    avg_loss = total_loss / num_batches
    loss_dict = {
        'total': avg_loss,
        'mse': total_mse / num_batches,
        'physics': total_physics / num_batches,
        'chapman': total_chapman / num_batches,
        'tec_align': total_tec_align / num_batches
    }

    return avg_loss, loss_dict


def validate(model, val_loader, batch_processor, device, config):
    """
    验证模型

    Args:
        model: R-STMRF 模型
        val_loader: 验证数据 loader
        batch_processor: 批次处理器
        device: 设备
        config: 配置字典

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
            # 处理批次
            coords, target_ne, sw_seq, tec_map_seq, target_tec_map = batch_processor.process_batch(batch_data)

            # 前向传播
            pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_map_seq)

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

    # ==================== 3. 准备数据集 ====================
    print("\n[步骤 3] 准备数据集...")

    train_loader, val_loader = get_dataloaders(
        npy_path=config['fy_path'],
        val_days=config['val_days'],
        batch_size=config['batch_size'],
        bin_size_hours=config['bin_size_hours'],
        num_workers=config['num_workers']
    )

    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")

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
        config=config
    ).to(device)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

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
            model, train_loader, batch_processor, optimizer, device, config, epoch
        )
        train_losses.append(train_loss)

        # 验证
        val_loss, val_metrics = validate(model, val_loader, batch_processor, device, config)
        val_losses.append(val_loss)

        # 打印结果
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"    - MSE: {train_dict['mse']:.6f}")
        print(f"    - Physics: {train_dict['physics']:.6f}")
        print(f"      · Chapman: {train_dict['chapman']:.6f}")
        print(f"      · TEC Align: {train_dict['tec_align']:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"    - MAE: {val_metrics['mae']:.6f}")
        print(f"    - RMSE: {val_metrics['rmse']:.6f}")
        print(f"    - R²: {val_metrics['r2']:.4f}")

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

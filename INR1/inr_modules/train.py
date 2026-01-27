"""
训练脚本
完整的训练流程
"""
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from inr_modules.config import get_config
from inr_modules.data_managers import SpaceWeatherManager, TECDataManager
from inr_modules.models import PhysicsGuidedINR
from inr_modules.losses import (
    heteroscedastic_loss,
    iri_gradient_direction_loss,
    tec_gradient_alignment_loss,
    smoothness_loss_tv,
    get_obs_weight
)
from inr_modules.data_managers import IRINeuralProxy
from inr_modules.data_managers import get_dataloaders


def train_model(config):
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
    print(f"\n{'='*60}")
    print(f"使用设备: {device}")
    print(f"{'='*60}\n")
    
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
    
    # ==================== 2. 加载IRI神经代理 ====================
    print("\n[步骤 2] 加载IRI神经代理场...")
    
    if not os.path.exists(config['iri_proxy_path']):
        raise FileNotFoundError(f"IRI代理未找到: {config['iri_proxy_path']}")
    
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    state_dict = torch.load(config['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(state_dict)
    iri_proxy.eval()
    print("  IRI代理已加载并冻结")
    
    # ==================== 3. 准备数据集 ====================
    print("\n[步骤 3] 准备数据集（随机划分）...")
    
    # 加载全部数据
    full_loader_temp, _ = get_dataloaders(
        npy_path=config['fy_path'],
        val_days=[],  # 不使用日期过滤
        batch_size=config['batch_size'],
        bin_size_hours=config['time_res'],
        num_workers=0
    )
    
    # 随机划分
    full_dataset = full_loader_temp.dataset
    total_samples = len(full_dataset)
    val_size = int(total_samples * config['val_ratio'])
    train_size = total_samples - val_size
    
    print(f"  总样本: {total_samples}")
    print(f"  训练集: {train_size} | 验证集: {val_size}")
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"  训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}")
    
    # ==================== 4. 初始化模型 ====================
    print("\n[步骤 4] 初始化模型...")
    
    model = PhysicsGuidedINR(
        iri_proxy=iri_proxy,
        lat_range=config['lat_range'],
        lon_range=config['lon_range'],
        alt_range=config['alt_range'],
        sw_manager=sw_manager,
        tec_manager=tec_manager,
        start_date_str=config['start_date_str'],
        config=config
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # ==================== 5. 优化器和调度器 ====================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    
    # ==================== 6. 训练循环 ====================
    print("\n[步骤 5] 开始训练...")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  损失权重: IRI_Dir={config['w_iri_dir']}, "
          f"TEC_Align={config['w_tec_align']}, "
          f"Smooth={config['w_smooth']}\n")
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # ----- 训练阶段 -----
        model.train()
        train_loss_epoch = 0.0
        steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
    
            # ✅ 确保coords需要梯度
            batch_coords = batch[:, :4].clone().detach().requires_grad_(True)
            batch_y = batch[:, 4].unsqueeze(1)
    
          # 获取序列（这些不需要梯度）
            obs_sw_seq = sw_manager.get_drivers_sequence(batch_coords[:, 3])
            coords_no_grad = batch_coords.detach()
            obs_tec_seq = tec_manager.get_tec_sequence(
                coords_no_grad[:, 0], coords_no_grad[:, 1], coords_no_grad[:, 3]
            )
    
            optimizer.zero_grad()
    
            # 前向传播
            ne_pred, log_var, correction, delta_w, gamma_v, beta_v = model(
                batch_coords, obs_sw_seq, obs_tec_seq
            )
    
            # ✅ 获取IRI背景（会保留对batch_coords的梯度）
            iri_bg_val = model.get_background(
                batch_coords[:,0], batch_coords[:,1], 
                batch_coords[:,2], batch_coords[:,3]
            )
            
            # --- 损失计算 ---
            # 1. 观测数据拟合
            w_obs = get_obs_weight(batch_coords[:, 2])
            loss_obs = heteroscedastic_loss(ne_pred, batch_y, log_var, obs_weight=w_obs)
            
            # 2. IRI结构主导
            loss_iri_dir = iri_gradient_direction_loss(ne_pred, iri_bg_val, batch_coords)
            
            # 3. TEC水平调制
            loss_tec_align = tec_gradient_alignment_loss(ne_pred, batch_coords, tec_manager)
            
            # 4. 辅助平滑
            loss_smooth = smoothness_loss_tv(ne_pred, batch_coords)
            
            # 5. 正则化
            loss_reg = torch.mean(correction ** 2) * config['w_bkg_val']
            loss_mod_reg = (torch.mean(delta_w**2) + 
                          torch.mean(gamma_v**2) + 
                          torch.mean(beta_v**2))
            
            # 总损失
            loss = (loss_obs +
                   config['w_iri_dir'] * loss_iri_dir +
                   config['w_tec_align'] * loss_tec_align +
                   config['w_smooth'] * loss_smooth +
                   loss_reg + 0.05 * loss_mod_reg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += loss.item()
            steps += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{config['epochs']}] "
                      f"Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"(Obs: {loss_obs.item():.4f}, "
                      f"IRI-Dir: {loss_iri_dir.item():.4f}, "
                      f"TEC-Align: {loss_tec_align.item():.4f})")
        
        avg_train_loss = train_loss_epoch / steps if steps > 0 else 0
        train_losses.append(avg_train_loss)
        
        # ----- 验证阶段 -----
        model.eval()
        val_loss_epoch = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_coords = batch[:, :4]
                val_y = batch[:, 4].unsqueeze(1)
                
                val_sw_seq = sw_manager.get_drivers_sequence(val_coords[:, 3])
                val_tec_seq = tec_manager.get_tec_sequence(
                    val_coords[:, 0], val_coords[:, 1], val_coords[:, 3]
                )
                
                ne_pred, log_var, _, _, _, _ = model(
                    val_coords, val_sw_seq, val_tec_seq
                )
                
                w_obs = get_obs_weight(val_coords[:, 2])
                loss_val_batch = heteroscedastic_loss(
                    ne_pred, val_y, log_var, obs_weight=w_obs
                )
                
                val_loss_epoch += loss_val_batch.item()
                val_steps += 1
        
        avg_val_loss = val_loss_epoch / val_steps if val_steps > 0 else 0
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f"==> Epoch {epoch+1}/{config['epochs']} "
              f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(config['save_dir'], exist_ok=True)
            save_path = os.path.join(config['save_dir'], 'best_model_vtec_mod.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  已保存最佳模型: {save_path}")
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}\n")
    
    return model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager


if __name__ == '__main__':
    config = get_config()
    
    model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager = train_model(config)
    
    print("训练完成，模型和损失曲线已保存")

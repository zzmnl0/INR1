"""
高度剖面可视化工具
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_full_altitude_profile(model, sw_manager, tec_manager, device,
                               target_day, target_hour, save_dir):
    """
    绘制全高度层切片图
    
    Args:
        model: 训练好的模型
        sw_manager: 空间天气管理器
        tec_manager: TEC管理器
        device: 计算设备
        target_day: 目标日期
        target_hour: 目标小时
        save_dir: 保存目录
    """
    model.eval()
    global_time = target_day * 24.0 + target_hour
    
    # 获取SW序列（全图共享）
    time_tensor = torch.tensor([global_time], device=device)
    sw_seq = sw_manager.get_drivers_sequence(time_tensor)
    
    # 提取当前驱动值用于标题
    kp_curr = sw_seq[0, -1, 0].item()
    f107_curr = sw_seq[0, -1, 1].item()
    
    kp_disp = (kp_curr + 1.0) / 2.0 * 9.0
    f107_disp = f107_curr * 60.0 + 210.0
    
    print(f"绘制剖面 Day {target_day} H{target_hour:02.0f}. "
          f"驱动: Kp={kp_disp:.1f}, F10.7={f107_disp:.1f}")
    
    # 定义高度层
    alt_levels = list(range(120, 500, 30))
    num_alts = len(alt_levels)
    
    # 定义空间网格
    lat_grid = np.linspace(-90, 90, 91)
    lon_grid = np.linspace(-180, 180, 180)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    extent = [-180, 180, -90, 90]
    
    # 创建图形
    fig, axes = plt.subplots(nrows=num_alts, ncols=3, 
                            figsize=(18, 2.5 * num_alts))
    
    for i, alt in enumerate(alt_levels):
        ax_row = axes[i]
        
        # 构建查询坐标
        lat_flat = LAT.flatten()
        lon_flat = LON.flatten()
        alt_flat = np.full_like(lat_flat, alt)
        time_flat = np.full_like(lat_flat, global_time)
        
        coords = np.stack([lat_flat, lon_flat, alt_flat, time_flat], axis=1).astype(np.float32)
        coords_t = torch.from_numpy(coords).to(device)
        batch_len = coords_t.shape[0]
        
        # 构建输入序列
        sw_t = sw_seq.expand(batch_len, -1, -1)
        tec_t = tec_manager.get_tec_sequence(
            coords_t[:, 0], coords_t[:, 1], coords_t[:, 3]
        )
        
        # 推理
        with torch.no_grad():
            iri_bg = model.get_background(
                coords_t[:, 0], coords_t[:, 1], coords_t[:, 2], coords_t[:, 3]
            ).squeeze().cpu().numpy()
            
            ne_pred, _, _, _, _, _ = model(coords_t, sw_t, tec_t)
            inr_pred = ne_pred.squeeze().cpu().numpy()
            
            iri_map = iri_bg.reshape(LAT.shape)
            inr_map = inr_pred.reshape(LAT.shape)
            res_map = inr_map - iri_map
        
        # 确定颜色范围
        local_min = max(min(np.nanmin(iri_map), np.nanmin(inr_map)), 8.0)
        local_max = min(max(np.nanmax(iri_map), np.nanmax(inr_map)), 13.0)
        
        if local_max - local_min < 0.2:
            mid = (local_max + local_min) / 2
            local_max = mid + 0.1
            local_min = mid - 0.1
        
        local_res_max = max(np.nanmax(np.abs(res_map)), 0.2)
        
        # 绘制三列
        im1 = ax_row[0].imshow(iri_map, extent=extent, origin='lower',
                               cmap='plasma', vmin=local_min, vmax=local_max,
                               aspect='auto')
        ax_row[0].set_ylabel(f'{alt} km\nLat', fontweight='bold')
        if i == 0:
            ax_row[0].set_title('IRI Background', fontsize=12)
        plt.colorbar(im1, ax=ax_row[0], pad=0.02)
        
        im2 = ax_row[1].imshow(inr_map, extent=extent, origin='lower',
                               cmap='plasma', vmin=local_min, vmax=local_max,
                               aspect='auto')
        if i == 0:
            ax_row[1].set_title('INR Prediction (VTEC+SW)', fontsize=12)
        plt.colorbar(im2, ax=ax_row[1], pad=0.02)
        
        im3 = ax_row[2].imshow(res_map, extent=extent, origin='lower',
                               cmap='bwr', vmin=-local_res_max, vmax=local_res_max,
                               aspect='auto')
        if i == 0:
            ax_row[2].set_title('Residual', fontsize=12)
        plt.colorbar(im3, ax=ax_row[2], pad=0.02)
        
        # 设置x轴标签
        if i < num_alts - 1:
            for ax in ax_row:
                ax.set_xticks([])
        else:
            for ax in ax_row:
                ax.set_xlabel('Longitude', fontsize=10)
    
    # 总标题
    plt.suptitle(
        f'重构结果 Day {target_day}, {target_hour:02.0f}:00 UT '
        f'(Kp={kp_disp:.1f}, F10.7={f107_disp:.1f})\n'
        f'模式: VTEC空间调制 + SW弱调制',
        fontsize=14, fontweight='bold', y=1.01
    )
    
    plt.tight_layout()
    
    filename = f'profile_day{target_day:02d}_hour{int(target_hour):02d}_vtec_mod.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"剖面已保存: {save_path}")

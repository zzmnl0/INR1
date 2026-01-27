"""
评估与可视化工具
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import os


def evaluate_and_save_report(model, train_loader, val_loader, 
                             sw_manager, tec_manager, device, save_dir):
    """
    计算详细评估指标并保存报告
    
    Args:
        model: 训练好的模型
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        sw_manager: 空间天气管理器
        tec_manager: TEC管理器
        device: 计算设备
        save_dir: 保存目录
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    def get_metrics(dataloader, name):
        """计算单个数据集的指标"""
        print(f"评估 {name} 集指标...")
        obs_list, pred_list = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                coords = batch[:, :4]
                y_true = batch[:, 4]
                
                sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
                tec_seq = tec_manager.get_tec_sequence(
                    coords[:, 0], coords[:, 1], coords[:, 3]
                )
                
                ne_pred, _, _, _, _, _ = model(coords, sw_seq, tec_seq)
                
                obs_list.append(y_true.cpu().numpy())
                pred_list.append(ne_pred.squeeze().cpu().numpy())
        
        obs = np.concatenate(obs_list)
        pred = np.concatenate(pred_list)
        
        rmse = np.sqrt(mean_squared_error(obs, pred))
        r2 = r2_score(obs, pred)
        r_corr, _ = pearsonr(obs, pred)
        
        return rmse, r2, r_corr, len(obs)
    
    # 计算指标
    t_rmse, t_r2, t_r, t_len = get_metrics(train_loader, "训练")
    v_rmse, v_r2, v_r, v_len = get_metrics(val_loader, "验证")
    
    # 生成报告
    content = []
    content.append("=" * 60)
    content.append("   电离层重构评估报告")
    content.append("=" * 60)
    content.append(f"模型: Physics-Guided INR (VTEC+SW)")
    content.append(f"划分策略: 随机划分 (训练 90% / 验证 10%)")
    content.append(f"总样本数: {t_len + v_len} (训练: {t_len}, 验证: {v_len})")
    content.append("-" * 60)
    content.append(f"[训练集指标]")
    content.append(f"  RMSE : {t_rmse:.5f}")
    content.append(f"  R^2  : {t_r2:.5f}")
    content.append(f"  R    : {t_r:.5f}")
    content.append("-" * 60)
    content.append(f"[验证集指标]")
    content.append(f"  RMSE : {v_rmse:.5f}")
    content.append(f"  R^2  : {v_r2:.5f}")
    content.append(f"  R    : {v_r:.5f}")
    content.append("=" * 60)
    
    report_text = "\n".join(content)
    print(report_text)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"报告已保存: {report_path}")


def plot_loss_curve(train_losses, val_losses, save_path):
    """
    绘制损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def evaluate_parity(model, dataloader, sw_manager, tec_manager, 
                    device, save_dir, config):
    """
    绘制Parity Plot（散点图和密度图）
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        sw_manager: 空间天气管理器
        tec_manager: TEC管理器
        device: 计算设备
        save_dir: 保存目录
        config: 配置字典
    """
    model.eval()
    obs, iri, inr = [], [], []
    
    print("生成Parity图 (完整验证集)...")
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            coords = batch[:, :4]
            y = batch[:, 4]
            
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            tec_seq = tec_manager.get_tec_sequence(
                coords[:, 0], coords[:, 1], coords[:, 3]
            )
            
            iri_bg = model.get_background(
                coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            ).squeeze()
            
            ne_pred, _, _, _, _, _ = model(coords, sw_seq, tec_seq)
            inr_pred = ne_pred.squeeze()
            
            obs.append(y.cpu().numpy())
            iri.append(iri_bg.cpu().numpy())
            inr.append(inr_pred.cpu().numpy())
    
    obs = np.concatenate(obs)
    iri = np.concatenate(iri)
    inr = np.concatenate(inr)
    
    print(f"  处理点数: {len(obs)}")
    
    # 计算指标
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(((y_true - y_pred)**2).mean())
        r2 = r2_score(y_true, y_pred)
        r_corr, _ = pearsonr(y_true, y_pred)
        return rmse, r2, r_corr
    
    rmse_iri, r2_iri, r_iri = calc_metrics(obs, iri)
    rmse_inr, r2_inr, r_inr = calc_metrics(obs, inr)
    
    val_days_str = str(config.get('val_days', '随机划分'))
    
    # 确定坐标轴范围
    global_min = min(obs.min(), iri.min(), inr.min())
    global_max = max(obs.max(), iri.max(), inr.max())
    axis_min = np.floor(global_min * 10) / 10
    axis_max = np.ceil(global_max * 10) / 10
    
    title_iri = (f'IRI-2020 vs 观测 (划分: {val_days_str})\n'
                 f'RMSE={rmse_iri:.4f} | R²={r2_iri:.4f} | R={r_iri:.4f}')
    title_inr = (f'Physics-Guided INR (VTEC Mod) vs 观测\n'
                 f'RMSE={rmse_inr:.4f} | R²={r2_inr:.4f} | R={r_inr:.4f}')
    
    # 绘制散点图
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    def draw_scatter(axis, x, y, title_text, color):
        axis.scatter(x, y, alpha=0.05, s=0.5, c=color, rasterized=True)
        axis.plot([axis_min, axis_max], [axis_min, axis_max], 
                 'r--', lw=2, alpha=0.8, label='1:1 Line')
        axis.set_title(title_text, fontsize=12, fontweight='bold')
        axis.set_xlabel('观测 Ne (log10)', fontsize=11)
        axis.set_ylabel('模型 Ne (log10)', fontsize=11)
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_aspect('equal')
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)
    
    draw_scatter(ax1[0], obs, iri, title_iri, 'blue')
    draw_scatter(ax1[1], obs, inr, title_inr, 'green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parity_scatter_vtec.png'), dpi=150)
    plt.close(fig1)
    
    # 绘制密度图
    fig2, ax2 = plt.subplots(1, 2, figsize=(17, 7), dpi=150)
    plot_range = [[axis_min, axis_max], [axis_min, axis_max]]
    bins = 400
    
    def draw_density(axis, x, y, title_text):
        h = axis.hist2d(x, y, bins=bins, range=plot_range, 
                       cmap='turbo', norm=LogNorm(), cmin=1)
        axis.plot([axis_min, axis_max], [axis_min, axis_max], 
                 'r--', lw=1.5, alpha=0.8, label='1:1 Line')
        axis.set_title(title_text, fontsize=12, fontweight='bold')
        axis.set_xlabel('观测 Ne (log10)', fontsize=11)
        axis.set_ylabel('模型 Ne (log10)', fontsize=11)
        axis.set_aspect('equal')
        axis.grid(True, linestyle=':', alpha=0.4)
        axis.legend(fontsize=10)
        
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(h[3], cax=cax, label='计数 (对数尺度)')
    
    draw_density(ax2[0], obs, iri, title_iri)
    draw_density(ax2[1], obs, inr, title_inr)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parity_density_vtec.png'), dpi=150)
    plt.close(fig2)
    
    print("Parity图已保存")

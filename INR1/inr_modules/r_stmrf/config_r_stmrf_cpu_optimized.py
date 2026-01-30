"""
R-STMRF 模型配置文件 - CPU 内存优化版本

针对 CPU 环境（8GB 可用内存）的优化配置
预期内存: ~200-300 MB per batch
预期磁盘: ~5 MB
"""

import torch
import os

CONFIG_R_STMRF_CPU_OPTIMIZED = {
    # ==================== 数据路径 ====================
    'fy_path': r'D:\FYsatellite\EDP_data\fy_202409_clean.npy',
    'iri_proxy_path': r"D:\code11\IRI01\output_results\iri_september_full_proxy.pth",
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'tec_path': r'D:\IGS\VTEC\tec_map_data.npy',
    'save_dir': './checkpoints_r_stmrf_cpu',

    # ==================== 数据规格 ====================
    'total_hours': 720.0,
    'start_date_str': '2024-09-01 00:00:00',
    'time_res': 3.0,
    'bin_size_hours': 3.0,

    # ==================== 物理参数 ====================
    'lat_range': (-90.0, 90.0),
    'lon_range': (-180.0, 180.0),
    'alt_range': (120.0, 500.0),

    # ==================== 时序学习参数 ====================
    'seq_len': 4,  # 6 → 4 (减少33%内存)

    # ==================== SIREN 架构参数（降低维度）====================
    'basis_dim': 48,  # 64 → 48 (减少25%)
    'siren_hidden': 96,  # 128 → 96 (减少25%)
    'siren_layers': 2,  # 3 → 2 (减少1层)
    'omega_0': 30.0,

    # ==================== 循环网络参数 ====================
    # TEC 参数（保持原始分辨率，仅减少通道数）
    # 原始分辨率: 73×73 (纬度填充后)
    'tec_feat_dim': 12,  # 16 → 12 (减少25%)
    'tec_h': 73,  # 原始分辨率（纬度填充后）
    'tec_w': 73,  # 原始分辨率
    'convlstm_layers': 1,
    'convlstm_kernel': 3,

    # LSTM (全局环境编码器 - 轻量化)
    'env_hidden_dim': 48,  # 64 → 48
    'lstm_layers': 2,
    'lstm_dropout': 0.1,

    # ==================== 训练超参数 ====================
    'batch_size': 512,  # 2048 → 512 (减少75%内存)
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 30,  # 50 → 30 (减少训练时间)
    'seed': 42,
    'device': 'cpu',  # 强制CPU
    'num_workers': 0,

    # ==================== 学习率调度 ====================
    'scheduler_type': 'cosine',
    'warmup_epochs': 2,  # 3 → 2
    'min_lr': 1e-6,

    # ==================== 数据划分 ====================
    'val_days': [],
    'val_ratio': 0.1,

    # ==================== 损失函数权重 ====================
    'w_mse': 1.0,
    'w_chapman': 0.1,
    'w_tec_direction': 0.02,  # 0.03 → 0.02 (降低约束强度)
    'w_tec_align': 0.0,
    'w_smooth': 0.0,
    'w_iri_dir': 0.0,
    'w_bkg_val': 0.0,

    # ==================== 不确定性学习 ====================
    'use_uncertainty': True,
    'uncertainty_weight': 0.5,

    # ==================== 模型保存（减少磁盘占用）====================
    'save_interval': 10,  # 5 → 10 (减少保存频率)
    'save_best_only': True,  # 只保存最佳模型

    # ==================== 可视化 ====================
    'plot_interval': 20,  # 减少可视化频率
    'plot_days': [15],  # 只可视化1天
    'plot_hours': [0.0, 12.0],  # 减少时刻数

    # ==================== 早停 ====================
    'early_stopping': True,
    'patience': 8,  # 10 → 8

    # ==================== 梯度裁剪 ====================
    'grad_clip': 1.0,

    # ==================== 混合精度训练 ====================
    'use_amp': False,  # CPU 不支持

    # ==================== TEC 梯度对齐参数 ====================
    'tec_gradient_threshold_percentile': 50.0,

    # ==================== 多时间尺度优化（CPU推荐启用）====================
    'use_tec_cache': True,  # 启用小时级TEC缓存（减少ConvLSTM计算次数）
    'tec_cache_size': 50,  # CPU环境使用较小的缓存
}


def get_config_r_stmrf_cpu_optimized():
    """获取 CPU 优化配置字典"""
    os.makedirs(CONFIG_R_STMRF_CPU_OPTIMIZED['save_dir'], exist_ok=True)
    return CONFIG_R_STMRF_CPU_OPTIMIZED


def print_memory_estimate():
    """打印内存估算"""
    config = CONFIG_R_STMRF_CPU_OPTIMIZED

    # 模型参数估算
    basis_dim = config['basis_dim']
    siren_hidden = config['siren_hidden']
    tec_feat_dim = config['tec_feat_dim']
    tec_h, tec_w = config['tec_h'], config['tec_w']

    # 粗略估算
    model_params = (basis_dim * siren_hidden * 2 +  # SIREN
                    tec_feat_dim * tec_h * tec_w * 2 +  # ConvLSTM
                    48 * 64 * 2)  # LSTM

    model_memory_mb = model_params * 4 / (1024 ** 2)  # FP32

    # 前向传播内存
    batch_size = config['batch_size']
    forward_memory_mb = (
        batch_size * basis_dim * 2 * 4 / (1024 ** 2) +  # h_spatial, h_temporal
        100 * tec_feat_dim * tec_h * tec_w * 4 / (1024 ** 2)  # 假设100个唯一窗口
    )

    total_memory_mb = model_memory_mb * 3 + forward_memory_mb * 2  # 粗略估算

    print("\n" + "="*70)
    print("CPU 优化配置 - 内存估算")
    print("="*70)
    print(f"\n模型参数内存: ~{model_memory_mb:.1f} MB")
    print(f"前向传播内存: ~{forward_memory_mb:.1f} MB")
    print(f"预计峰值内存: ~{total_memory_mb:.0f} MB")
    print(f"\n推荐系统内存: >= 2 GB")
    print(f"实际可用内存: 建议 >= 4 GB")
    print("="*70 + "\n")


def compare_with_default():
    """与默认配置对比"""
    default_config = {
        'batch_size': 2048,
        'seq_len': 6,
        'basis_dim': 64,
        'siren_hidden': 128,
        'tec_downsample_factor': 4,
        'tec_feat_dim': 16,
        'epochs': 50,
    }

    optimized = CONFIG_R_STMRF_CPU_OPTIMIZED

    print("\n" + "="*70)
    print("配置对比：默认 vs CPU优化")
    print("="*70)
    print(f"\n{'参数':<20} {'默认':<15} {'CPU优化':<15} {'减少':<10}")
    print("-"*70)

    params = [
        ('batch_size', 'batch_size', ''),
        ('seq_len', 'seq_len', ''),
        ('basis_dim', 'basis_dim', ''),
        ('siren_hidden', 'siren_hidden', ''),
        ('tec_feat_dim', 'tec_feat_dim', ''),
        ('epochs', 'epochs', ''),
    ]

    for name, key, unit in params:
        default_val = default_config.get(key, '-')
        optimized_val = optimized.get(key, '-')
        if isinstance(default_val, (int, float)) and isinstance(optimized_val, (int, float)):
            reduction = f"{(1 - optimized_val/default_val)*100:.0f}%"
        else:
            reduction = '-'
        print(f"{name:<20} {str(default_val):<15} {str(optimized_val):<15} {reduction:<10}")

    # 估算内存减少
    memory_reduction = (1 - (512 * 48 * 96 * 12) / (2048 * 64 * 128 * 16)) * 100
    print("\n" + "-"*70)
    print(f"{'估算内存减少':<20} {'~700 MB':<15} {'~250 MB':<15} {f'{memory_reduction:.0f}%':<10}")
    print("="*70 + "\n")


if __name__ == '__main__':
    print("\n🚀 R-STMRF CPU 优化配置")
    print_memory_estimate()
    compare_with_default()

    print("\n📝 使用方法:")
    print("```python")
    print("from config_r_stmrf_cpu_optimized import get_config_r_stmrf_cpu_optimized")
    print("config = get_config_r_stmrf_cpu_optimized()")
    print("```")
    print("\n或修改 config_r_stmrf.py 中的参数\n")

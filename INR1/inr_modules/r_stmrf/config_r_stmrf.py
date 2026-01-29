"""
R-STMRF 模型配置文件

包含所有训练和模型的全局配置参数
"""

import torch
import os

CONFIG_R_STMRF = {
    # ==================== 数据路径 ====================
    'fy_path': r'D:\FYsatellite\EDP_data\fy_202409_clean.npy',
    'iri_proxy_path': r"D:\code11\IRI01\output_results\iri_september_full_proxy.pth",
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'tec_path': r'D:\IGS\VTEC\tec_map_data.npy',
    'save_dir': './checkpoints_r_stmrf',

    # ==================== 数据规格 ====================
    'total_hours': 720.0,  # 总时长（小时）
    'start_date_str': '2024-09-01 00:00:00',  # 起始时间
    'time_res': 3.0,  # IRI 时间分辨率（小时）
    'bin_size_hours': 3.0,  # 时间分箱大小（小时）

    # ==================== 物理参数 ====================
    'lat_range': (-90.0, 90.0),  # 纬度范围
    'lon_range': (-180.0, 180.0),  # 经度范围（注意：-180 与 180 重复）
    'alt_range': (120.0, 500.0),  # 高度范围（km）

    # ==================== 时序学习参数 ====================
    'seq_len': 6,  # 历史窗口长度（时间步）

    # ==================== SIREN 架构参数 ====================
    'basis_dim': 64,  # 空间/时间基函数维度
    'siren_hidden': 128,  # SIREN 隐层维度
    'siren_layers': 3,  # SIREN 隐层数量
    'omega_0': 30.0,  # SIREN 频率因子

    # ==================== 循环网络参数 ====================
    # ConvLSTM (TEC 区域级上下文编码器 - 梯度方向约束源，非数值调制)
    # 新设计: TEC 仅作为"时序水平梯度方向约束"，不直接调制电子密度
    # 职责: 提取 TEC 水平梯度方向的时序演化特征 F_TEC
    # 作用: 通过梯度方向一致性损失约束 Ne_fused，不参与数值预测
    #
    # TEC 数据规格（保持原始分辨率）:
    # - 原始: (720, 71, 73) - Time × Lat × Lon
    # - Lat: -87.5 ~ 87.5, 步长 2.5° (71个点)
    # - Lon: -180 ~ 180, 步长 5° (73个点)
    # - 纬度填充后: (720, 73, 73) - 填充到 -90 ~ 90
    #
    # 内存优化:
    # ConvLSTM 作用于区域级背景场，不随 batch_size 增长
    # 内存占用: N_unique * tec_feat_dim * tec_h * tec_w * 4 bytes（与 batch_size 无关！）
    #   示例: 原始分辨率 73×73
    #         10 * 16 * 73 * 73 * 4 = ~3.4 MB（极小，即使 N_unique=100 也只需 34 MB）
    #         100 * 16 * 73 * 73 * 4 = ~34 MB（可接受，batch_size=2048 无压力）
    'tec_feat_dim': 16,  # ConvLSTM 输出通道数（梯度方向特征，无需太大）
    'tec_h': 73,  # TEC 地图高度（纬度填充后）
    'tec_w': 73,  # TEC 地图宽度（原始）
    'convlstm_layers': 1,  # ConvLSTM 层数（简化，只需提取低频梯度方向）
    'convlstm_kernel': 3,  # ConvLSTM 卷积核大小

    # LSTM (全局环境编码器)
    'env_hidden_dim': 64,  # LSTM 隐层维度
    'lstm_layers': 2,  # LSTM 层数
    'lstm_dropout': 0.1,  # LSTM Dropout

    # ==================== 训练超参数 ====================
    'batch_size': 2048,  # 批次大小（ConvLSTM 不随 batch 增长，可使用大 batch）
    'lr': 3e-4,  # 学习率
    'weight_decay': 1e-4,  # 权重衰减
    'epochs': 50,  # 训练轮数
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,  # DataLoader 工作进程数

    # ==================== 学习率调度 ====================
    'scheduler_type': 'cosine',  # 学习率调度器类型 ['cosine', 'step', 'plateau']
    'warmup_epochs': 3,  # 预热轮数
    'min_lr': 1e-6,  # 最小学习率

    # ==================== 数据划分 ====================
    'val_days': [],  # 验证集日期（留空表示使用随机划分）
    'val_ratio': 0.1,  # 验证集比例（随机划分）

    # ==================== 损失函数权重 ====================
    # 主损失
    'w_mse': 1.0,  # MSE 损失权重（或 Huber Loss）

    # 物理约束损失（新设计）
    'w_chapman': 0.1,  # Chapman 垂直平滑损失权重
    'w_tec_direction': 0.03,  # TEC 梯度方向一致性权重（新设计 - 取较小值避免过约束）

    # 已弃用的损失（兼容性保留）
    'w_tec_align': 0.0,  # 旧的 TEC 梯度对齐损失（已弃用，设为 0）
    'w_smooth': 0.0,  # 额外的平滑约束（已弃用）
    'w_iri_dir': 0.0,  # IRI 梯度方向一致性（已弃用）
    'w_bkg_val': 0.0,  # 背景值正则化（已弃用）

    # ==================== 不确定性学习 ====================
    'use_uncertainty': True,  # 是否启用异方差损失
    'uncertainty_weight': 0.5,  # 不确定性项权重

    # ==================== 模型保存 ====================
    'save_interval': 5,  # 每隔多少个 epoch 保存一次模型
    'save_best_only': True,  # 是否只保存最佳模型

    # ==================== 可视化 ====================
    'plot_interval': 10,  # 每隔多少个 epoch 绘制可视化
    'plot_days': [5, 15, 25],  # 绘制哪些天的高度剖面
    'plot_hours': [0.0, 6.0, 12.0, 18.0],  # 绘制哪些时刻

    # ==================== 早停 ====================
    'early_stopping': True,  # 是否启用早停
    'patience': 10,  # 早停耐心值（验证损失不下降的轮数）

    # ==================== 梯度裁剪 ====================
    'grad_clip': 1.0,  # 梯度裁剪阈值（设为 None 则不裁剪）

    # ==================== 混合精度训练 ====================
    'use_amp': False,  # 是否使用自动混合精度（AMP）

    # ==================== TEC 梯度对齐参数 ====================
    'tec_gradient_threshold_percentile': 50.0,  # TEC 梯度显著性阈值（百分位数）
}


def get_config_r_stmrf():
    """获取 R-STMRF 配置字典"""
    # 确保保存目录存在
    os.makedirs(CONFIG_R_STMRF['save_dir'], exist_ok=True)
    return CONFIG_R_STMRF


def update_config_r_stmrf(**kwargs):
    """更新配置参数"""
    CONFIG_R_STMRF.update(kwargs)


def print_config_r_stmrf():
    """打印当前配置"""
    print("\n" + "="*70)
    print("R-STMRF 配置参数")
    print("="*70)

    # 分类打印
    categories = {
        '数据路径': ['fy_path', 'iri_proxy_path', 'sw_path', 'tec_path', 'save_dir'],
        '数据规格': ['total_hours', 'start_date_str', 'time_res', 'bin_size_hours'],
        '物理参数': ['lat_range', 'lon_range', 'alt_range'],
        '时序参数': ['seq_len'],
        'SIREN 架构': ['basis_dim', 'siren_hidden', 'siren_layers', 'omega_0'],
        '循环网络': ['tec_feat_dim', 'tec_h', 'tec_w', 'convlstm_layers', 'convlstm_kernel',
                      'env_hidden_dim', 'lstm_layers', 'lstm_dropout'],
        '训练超参数': ['batch_size', 'lr', 'weight_decay', 'epochs', 'device', 'num_workers'],
        '学习率调度': ['scheduler_type', 'warmup_epochs', 'min_lr'],
        '数据划分': ['val_days', 'val_ratio'],
        '损失权重': ['w_mse', 'w_chapman', 'w_tec_align', 'w_smooth', 'w_iri_dir', 'w_bkg_val',
                     'use_uncertainty', 'uncertainty_weight'],
        '其他': ['save_interval', 'save_best_only', 'plot_interval', 'early_stopping',
                 'patience', 'grad_clip', 'use_amp'],
    }

    for category, keys in categories.items():
        print(f"\n【{category}】")
        for key in keys:
            if key in CONFIG_R_STMRF:
                value = CONFIG_R_STMRF[key]
                print(f"  {key:30s}: {value}")

    print("\n" + "="*70 + "\n")


def validate_config():
    """验证配置合理性"""
    config = CONFIG_R_STMRF

    # 检查必要路径
    required_paths = ['fy_path', 'iri_proxy_path', 'sw_path', 'tec_path']
    for path_key in required_paths:
        path = config[path_key]
        if not os.path.exists(path):
            print(f"⚠️  警告: {path_key} 路径不存在: {path}")

    # 检查参数合理性
    assert config['seq_len'] > 0, "seq_len 必须大于 0"
    assert config['basis_dim'] > 0, "basis_dim 必须大于 0"
    assert config['batch_size'] > 0, "batch_size 必须大于 0"
    assert 0 < config['lr'] < 1, "学习率必须在 (0, 1) 之间"

    print("✅ 配置验证通过")


# ======================== 使用示例 ========================
if __name__ == '__main__':
    print_config_r_stmrf()
    validate_config()

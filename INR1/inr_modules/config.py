"""
配置文件模块
包含所有训练和模型的全局配置参数
"""
import torch
import os

CONFIG = {
    # ==================== 数据路径 ====================
    'fy_path': r'D:\FYsatellite\EDP_data\fy_202409_clean.npy',
    'iri_proxy_path': r"D:\code11\IRI01\output_results\iri_september_full_proxy.pth", 
    'sw_path': r'D:\FYsatellite\EDP_data\kp\OMNI_Kp_F107_20240901_20241001.txt',
    'tec_path': r'D:\IGS\VTEC\tec_map_data.npy', 
    'save_dir': './checkpoints',
    
    # ==================== 数据规格 ====================
    'total_hours': 720.0,  # 总时长（小时）
    'start_date_str': '2024-09-01 00:00:00',  # 起始时间
    'time_res': 3.0,  # IRI时间分辨率（小时）
    
    # ==================== 物理参数 ====================
    'lat_range': (-90.0, 90.0),  # 纬度范围
    'lon_range': (-180.0, 179.0),  # 经度范围
    'alt_range': (120.0, 500.0),  # 高度范围（km）
    
    # ==================== 时序学习参数 ====================
    'seq_len': 6,  # 历史窗口长度（时间步）
    'd_model': 64,  # Transformer隐层维度
    'nhead': 4,  # 多头注意力头数
    'num_layers': 2,  # Transformer编码器层数
    
    # ==================== 模型架构参数 ====================
    'basis_dim': 64,  # 空间基函数维度
    'siren_hidden': 128,  # SIREN隐层维度
    
    # ==================== 训练超参数 ====================
    'batch_size': 2048,
    'lr': 5e-4,  # 学习率
    'weight_decay': 1e-4,  # 权重衰减
    'epochs': 40,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,  # DataLoader工作进程数
    
    # ==================== 数据划分 ====================
    'val_days': [],  # 验证集日期（空则使用随机划分）
    'val_ratio': 0.1,  # 验证集比例
    
    # ==================== 损失函数权重 ====================
    'lambda_bkg': 0.05,  # 背景场约束权重
    'w_iri_dir': 0.2,  # IRI梯度方向一致性权重
    'w_tec_align': 0.05,  # TEC梯度对齐权重
    'w_smooth': 0.1,  # 平滑约束权重
    'w_bkg_val': 0.02,  # 背景值正则化权重
}


def get_config():
    """获取配置字典"""
    # 确保保存目录存在
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    return CONFIG


def update_config(**kwargs):
    """更新配置参数"""
    CONFIG.update(kwargs)


def print_config():
    """打印当前配置"""
    print("\n" + "="*60)
    print("当前配置参数:")
    print("="*60)
    for key, value in CONFIG.items():
        print(f"{key:25s}: {value}")
    print("="*60 + "\n")

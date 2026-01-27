"""
物理约束损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def heteroscedastic_loss(pred, target, log_var, obs_weight=None):
    """
    异方差损失函数 (Negative Log Likelihood)
    用于不确定性学习
    
    Loss = 0.5 * exp(-s) * (y - y_hat)^2 + 0.5 * s
    
    Args:
        pred: 模型预测均值 [Batch, 1]
        target: 观测真值 [Batch, 1]
        log_var: 模型预测的对数方差 [Batch, 1]
        obs_weight: 外部赋予的观测权重 [Batch, 1]
        
    Returns:
        scalar loss
    """
    precision = torch.exp(-log_var)
    mse_term = (pred - target) ** 2
    
    loss_element = 0.5 * precision * mse_term + 0.5 * log_var
    
    if obs_weight is not None:
        loss_element = loss_element * obs_weight
    
    return torch.mean(loss_element)


def iri_gradient_direction_loss(pred_ne, background_ne, coords):
    """
    IRI梯度方向一致性损失
    
    约束预测场与背景场的梯度方向一致，保留IRI的3D拓扑结构
    
    Args:
        pred_ne: 预测Ne值 [Batch, 1]
        background_ne: IRI背景值 [Batch, 1]
        coords: 坐标 [Batch, 4] (requires_grad=True)
        
    Returns:
        scalar loss
    """
    # 计算预测场梯度
    grad_pred = torch.autograd.grad(
        pred_ne, coords, 
        torch.ones_like(pred_ne),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # 计算IRI梯度
    grad_bkg = torch.autograd.grad(
        background_ne, coords,
        torch.ones_like(background_ne),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 关注3D方向 (Lat, Lon, Alt)
    grad_pred_3d = grad_pred[:, 0:3]
    grad_bkg_3d = grad_bkg[:, 0:3]
    
    # 余弦相似度
    cosine_sim = F.cosine_similarity(grad_pred_3d, grad_bkg_3d, dim=1, eps=1e-8)
    
    return 1.0 - torch.mean(cosine_sim)


def tec_gradient_alignment_loss(pred_ne, coords, tec_manager):
    """
    TEC梯度水平对齐损失
    
    约束预测场Ne的水平梯度与TEC地图的水平梯度一致
    
    Args:
        pred_ne: 预测Ne值 [Batch, 1]
        coords: 坐标 [Batch, 4] (requires_grad=True)
        tec_manager: TEC数据管理器
        
    Returns:
        scalar loss
    """
    # 1. 计算Ne对Lat/Lon的梯度
    grad_ne = torch.autograd.grad(
        outputs=pred_ne,
        inputs=coords,
        grad_outputs=torch.ones_like(pred_ne),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    vec_ne = grad_ne[:, 0:2]  # 水平分量
    
    # 2. 计算TEC对Lat/Lon的梯度
    tec_val = tec_manager.get_tec_sequence(
        coords[:, 0], coords[:, 1], coords[:, 3]
    )[:, -1, :]
    
    grad_tec = torch.autograd.grad(
        outputs=tec_val,
        inputs=coords,
        grad_outputs=torch.ones_like(tec_val),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 关键: 切断梯度链，防止反向传播到grid_sample
    vec_tec = grad_tec[:, 0:2].detach()
    
    # 3. 余弦相似度
    cosine_sim = F.cosine_similarity(vec_ne, vec_tec, dim=1, eps=1e-8)
    
    # 4. 掩码机制：只在TEC梯度显著处应用
    tec_grad_mag = torch.norm(vec_tec, dim=1)
    mask = (tec_grad_mag > tec_grad_mag.mean()).float()
    
    loss = 1.0 - (cosine_sim * mask).sum() / (mask.sum() + 1e-6)
    
    return loss


def smoothness_loss_tv(pred_ne, coords):
    """
    总变分平滑约束
    
    惩罚二阶导数，去除斑点，自发形成结构
    
    Args:
        pred_ne: 预测Ne值 [Batch, 1]
        coords: 坐标 [Batch, 4] (requires_grad=True)
        
    Returns:
        scalar loss
    """
    # 计算一阶导数
    grad = torch.autograd.grad(
        pred_ne, coords,
        torch.ones_like(pred_ne),
        create_graph=True
    )[0]
    
    d_lat = grad[:, 0]
    d_lon = grad[:, 1]
    d_alt = grad[:, 2]
    d_time = grad[:, 3]
    
    # 权重配置：
    # - 极大地惩罚时间导数 (Time=5.0)，强迫时间演化变得平滑
    # - 适度惩罚水平导数 (Lat/Lon=1.0)，消除斑点
    # - 弱惩罚垂直导数 (Alt=0.1)，保留Chapman结构
    loss = (1.0 * torch.mean(torch.abs(d_lat)**2) +
            1.0 * torch.mean(torch.abs(d_lon)**2) +
            0.1 * torch.mean(torch.abs(d_alt)**2) +
            5.0 * torch.mean(torch.abs(d_time)**2))
    
    return loss


def get_obs_weight(alt_batch):
    """
    高度相关观测权重 (R矩阵模拟)
    
    Args:
        alt_batch: [Batch] 高度值
        
    Returns:
        [Batch, 1] 权重
    """
    weights = torch.ones_like(alt_batch)
    
    # 低层权重降低
    weights[alt_batch < 100.0] = 0.1
    
    # 高层权重降低
    weights[alt_batch > 550.0] = 0.5
    
    return weights.unsqueeze(1)


def get_background_trust(lat_batch):
    """
    背景信任度 (B矩阵模拟)
    
    Args:
        lat_batch: [Batch] 纬度值
        
    Returns:
        [Batch, 1] 信任度
    """
    abs_lat = torch.abs(lat_batch)
    trust = torch.ones_like(abs_lat) * 0.3
    
    # 赤道区域信任度低
    trust[abs_lat < 20.0] = 0.1
    
    # 极区信任度低
    trust[abs_lat > 70.0] = 0.2
    
    return trust.unsqueeze(1)

# R-STMRF: 物理引导的循环时空调制残差场

**Recurrent Spatio-Temporal Modulated Residual Field for Ionospheric Electron Density Reconstruction**

---

## 📋 目录

- [概述](#概述)
- [核心架构](#核心架构)
- [文件结构](#文件结构)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [物理约束](#物理约束)
- [与原模型对比](#与原模型对比)
- [常见问题](#常见问题)

---

## 概述

R-STMRF 是对原有 Physics-Guided INR 模型的**重大升级**，核心改进包括：

### 核心创新

1. **TEC 作为空间上下文（Context）而非像素级输入**
   - 使用 **ConvLSTM** 提取 TEC 地图的时空演变特征
   - 通过 **FiLM 调制**约束水平梯度分布

2. **Kp/F10.7 作为时间调制器**
   - 使用 **LSTM** 编码全局环境状态
   - 通过 **加性调制（Additive Shift）**模拟磁暴期间的整体密度增益/衰减

3. **SIREN 基函数网络**
   - 替换 Fourier 特征编码为 **SIREN**（sin 激活 + 特殊初始化）
   - 更适合学习高频细节和周期性现象

4. **增强的物理约束**
   - **Chapman 垂直平滑损失**：约束高度方向二阶导数
   - **TEC 梯度对齐损失**：基于地图计算梯度方向一致性

---

## 核心架构

### 数学公式

```
Ne(x, t) = IRI_frozen(x, t) + Decoder(h_spatial_mod, h_temporal_mod)
```

其中：

#### 空间分支（Spatial Branch）
- **主路**: SIREN 空间基函数 → `h_spatial`
- **调制源**: ConvLSTM(TEC 地图序列) → 特征图 `F_tec`
- **调制方式**: FiLM → `h_spatial_mod = γ ⊙ h_spatial + β`

#### 时间分支（Temporal Branch）
- **主路**: SIREN 时间基函数 → `h_temporal`
- **调制源**: LSTM(Kp/F10.7 序列) → 全局状态 `z_env`
- **调制方式**: Additive Shift → `h_temporal_mod = h_temporal + β`

### 网络结构图

```
输入: (Lat, Lon, Alt, Time)
  │
  ├─ 空间路径 ────────────────────────┐
  │  · SIREN(Lat, Lon, Alt, sin_lt, cos_lt) → h_spatial
  │  · ConvLSTM(TEC Maps) → F_tec
  │  · grid_sample(F_tec, Lat, Lon) → z_tec
  │  · FiLM: γ, β ← MLP(z_tec)
  │  · h_spatial_mod = γ ⊙ h_spatial + β
  │
  ├─ 时间路径 ────────────────────────┤
  │  · SIREN(Time) → h_temporal
  │  · LSTM(Kp, F10.7) → z_env
  │  · Additive: β ← MLP(z_env)
  │  · h_temporal_mod = h_temporal + β
  │
  └─ 融合解码 ───────────────────────→
     · Decoder(Concat(h_spatial_mod, h_temporal_mod)) → Δlog Ne
     · Output = IRI_background + Δlog Ne
```

---

## 文件结构

```
INR1/inr_modules/r_stmrf/
├── __init__.py                       # 模块导出
├── siren_layers.py                   # SIREN 基础层
├── recurrent_parts.py                # LSTM + ConvLSTM 编码器
├── r_stmrf_model.py                  # 主模型
├── physics_losses_r_stmrf.py         # 物理约束损失
├── sliding_dataset.py                # 滑动窗口数据处理
├── config_r_stmrf.py                 # 配置文件
├── train_r_stmrf.py                  # 训练脚本
└── README_R_STMRF.md                 # 本文档

主入口:
INR1/main_r_stmrf.py                  # 主程序入口

数据管理器扩展:
INR1/inr_modules/data_managers/tec_manager.py
    └── 新增 get_tec_map_sequence() 方法
```

---

## 使用方法

### 1. 快速开始

```bash
# 切换到项目根目录
cd /path/to/INR1

# 运行训练
python main_r_stmrf.py
```

### 2. 自定义配置

编辑 `inr_modules/r_stmrf/config_r_stmrf.py`：

```python
CONFIG_R_STMRF = {
    # 数据路径
    'fy_path': 'path/to/fy_data.npy',
    'tec_path': 'path/to/tec_map_data.npy',

    # 模型超参数
    'basis_dim': 64,
    'siren_hidden': 128,
    'seq_len': 6,

    # 损失权重
    'w_chapman': 0.1,
    'w_tec_align': 0.05,

    # 训练参数
    'batch_size': 1024,
    'lr': 3e-4,
    'epochs': 50,
}
```

### 3. 仅训练模式

```python
from inr_modules.r_stmrf import train_r_stmrf, get_config_r_stmrf

config = get_config_r_stmrf()
model, train_losses, val_losses, *_ = train_r_stmrf(config)
```

### 4. 推理模式

```python
import torch
from inr_modules.r_stmrf import R_STMRF_Model

# 加载模型
model = R_STMRF_Model(...)
model.load_state_dict(torch.load('best_r_stmrf_model.pth'))
model.eval()

# 推理
with torch.no_grad():
    pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_map_seq)
```

---

## 配置说明

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seq_len` | 6 | 历史窗口长度（时间步） |
| `basis_dim` | 64 | 基函数维度 |
| `siren_hidden` | 128 | SIREN 隐层维度 |
| `siren_layers` | 3 | SIREN 隐层数量 |
| `omega_0` | 30.0 | SIREN 频率因子 |
| `tec_feat_dim` | 32 | ConvLSTM 输出通道数 |
| `env_hidden_dim` | 64 | LSTM 隐层维度 |

### 损失权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `w_mse` | 1.0 | MSE 损失权重 |
| `w_chapman` | 0.1 | Chapman 垂直平滑损失 |
| `w_tec_align` | 0.05 | TEC 梯度对齐损失 |
| `w_smooth` | 0.05 | 额外平滑约束（可选） |

---

## 物理约束

### 1. Chapman 垂直平滑损失

**物理意义**: Chapman 层的电子密度剖面应该平滑，无非物理震荡。

**实现**:
```python
def chapman_smoothness_loss(pred_ne, coords, alt_idx=2):
    # 计算二阶导数 ∂²Ne/∂h²
    grad_second = compute_second_derivative(pred_ne, coords, alt_idx)

    # 惩罚二阶导数
    loss = torch.mean(grad_second ** 2)
    return loss
```

**效果**:
- 抑制垂直方向的震荡
- 保持 Chapman 层的标准形态

### 2. TEC 梯度对齐损失

**物理意义**: TEC 是电子密度的垂直积分，水平梯度方向应一致。

**实现**:
```python
def tec_gradient_alignment_loss_v2(pred_ne, coords, target_tec_map):
    # 1. 计算 Ne 的水平梯度
    grad_ne = compute_horizontal_gradient(pred_ne, coords)

    # 2. 使用 Sobel 算子计算 TEC 地图梯度
    grad_tec = sobel_gradient(target_tec_map)

    # 3. 采样到查询点
    grad_tec_sampled = grid_sample(grad_tec, coords)

    # 4. 余弦相似度损失
    loss = 1 - cosine_similarity(grad_ne, grad_tec_sampled)
    return loss
```

**改进点**:
- 使用完整 TEC 地图（而非单点）
- Sobel 算子计算梯度
- 自适应掩码（只在梯度显著区域应用）

---

## 与原模型对比

| 特性 | 原 PhysicsGuidedINR | R-STMRF |
|------|---------------------|---------|
| **基函数** | Fourier Feature Encoding | **SIREN** (sin 激活) |
| **TEC 使用** | Transformer 单点序列 | **ConvLSTM 地图序列** |
| **调制方式** | FiLM (γ, β) | **Spatial: FiLM + Temporal: Additive** |
| **物理约束** | IRI 梯度 + TEC 单点对齐 | **Chapman 平滑 + TEC 地图对齐** |
| **参数量** | ~500K | ~800K (增加 ConvLSTM) |

### 优势

1. **更强的时空建模能力**
   - ConvLSTM 捕获 TEC 的水平梯度演变
   - LSTM 编码全局磁暴特征

2. **物理约束更精确**
   - Chapman 损失直接约束垂直形态
   - TEC 地图梯度提供更丰富的水平约束

3. **SIREN 优势**
   - 更适合学习高频细节
   - 梯度流动更稳定

---

## 常见问题

### Q1: TEC 数据格式要求？

**A**:
- 格式: `(T, 71, 73)` numpy array
- 自动上采样到 `(181, 361)` 用于 ConvLSTM
- 纬度: [-87.5, 87.5], 步长 2.5°
- 经度: [-180, 180], 步长 5°

### Q2: 显存不足怎么办？

**A**: 调整以下参数：
```python
'batch_size': 512,          # 减小批次大小
'tec_feat_dim': 16,         # 减少 ConvLSTM 通道数
'siren_hidden': 64,         # 减小 SIREN 隐层
```

### Q3: 训练不收敛？

**A**: 检查：
1. 损失权重是否合理（`w_chapman` 不要太大）
2. 学习率是否过大（建议 `1e-4 ~ 5e-4`）
3. 是否启用梯度裁剪（`grad_clip=1.0`）
4. TEC 数据是否正确归一化

### Q4: 如何可视化调制效果？

**A**:
```python
_, _, _, extras = model(coords, sw_seq, tec_map_seq)

# 查看调制参数
gamma = extras['gamma']              # 空间缩放
beta_spatial = extras['beta_spatial']  # 空间偏移
beta_temporal = extras['beta_temporal']  # 时间偏移

# 查看特征图
F_tec = extras['F_tec']  # TEC 特征图 [Batch, 32, 181, 361]
```

### Q5: 如何切换回原模型？

**A**:
```bash
# 使用原模型
python main_inr.py

# R-STMRF 模型
python main_r_stmrf.py
```

两个模型完全独立，可以同时保留。

---

## 引用

如果使用本模型，请引用：

```bibtex
@article{r_stmrf_2024,
  title={R-STMRF: Recurrent Spatio-Temporal Modulated Residual Field for Ionospheric Electron Density Reconstruction},
  author={Your Name},
  year={2024}
}
```

---

## 更新日志

### v1.0 (2024-XX-XX)
- ✅ 实现 SIREN 基函数网络
- ✅ 实现 ConvLSTM 空间上下文编码器
- ✅ 实现 LSTM 全局环境编码器
- ✅ 新增 Chapman 垂直平滑损失
- ✅ 改进 TEC 梯度对齐损失（基于地图）
- ✅ 完整训练和推理流程

---

## 联系方式

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Email: your-email@example.com

---

**Happy Coding! 🚀**

# R-STMRF 完整下载指南

## 🎯 快速下载

### 压缩包下载（一键获取所有文件）

```bash
# 压缩包位置
/home/user/INR1/INR1/r_stmrf_modules.tar.gz

# 解压到目标目录
tar -xzf r_stmrf_modules.tar.gz -C /your/target/directory/
```

---

## 📁 文件路径映射表

### 核心模块文件

| 本地路径 | 目标路径 | 必需 |
|---------|---------|------|
| `/home/user/INR1/INR1/inr_modules/r_stmrf/__init__.py` | `inr_modules/r_stmrf/__init__.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/siren_layers.py` | `inr_modules/r_stmrf/siren_layers.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/recurrent_parts.py` | `inr_modules/r_stmrf/recurrent_parts.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/r_stmrf_model.py` | `inr_modules/r_stmrf/r_stmrf_model.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/physics_losses_r_stmrf.py` | `inr_modules/r_stmrf/physics_losses_r_stmrf.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/sliding_dataset.py` | `inr_modules/r_stmrf/sliding_dataset.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/config_r_stmrf.py` | `inr_modules/r_stmrf/config_r_stmrf.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/train_r_stmrf.py` | `inr_modules/r_stmrf/train_r_stmrf.py` | ✅ |
| `/home/user/INR1/INR1/inr_modules/r_stmrf/README_R_STMRF.md` | `inr_modules/r_stmrf/README_R_STMRF.md` | 📖 |

### 主入口文件

| 本地路径 | 目标路径 | 必需 |
|---------|---------|------|
| `/home/user/INR1/INR1/main_r_stmrf.py` | `main_r_stmrf.py` | ✅ |

### 文档文件

| 本地路径 | 目标路径 | 必需 |
|---------|---------|------|
| `/home/user/INR1/INR1/R_STMRF_IMPLEMENTATION_SUMMARY.md` | `R_STMRF_IMPLEMENTATION_SUMMARY.md` | 📖 |
| `/home/user/INR1/INR1/FILE_MANIFEST.md` | `FILE_MANIFEST.md` | 📋 |

### 修改的现有文件

| 本地路径 | 目标路径 | 说明 |
|---------|---------|------|
| `/home/user/INR1/INR1/inr_modules/data_managers/tec_manager.py` | `inr_modules/data_managers/tec_manager.py` | 已添加 `get_tec_map_sequence()` 方法 |

---

## 📦 各文件功能速览

### 1. `siren_layers.py` - SIREN 基础层

**核心类**:
- `SIRENLayer`: 单层 SIREN (sin 激活)
- `SIRENNet`: 多层 SIREN 网络
- `ModulatedSIRENNet`: 可调制的 SIREN 网络

**使用示例**:
```python
from inr_modules.r_stmrf.siren_layers import SIRENNet

net = SIRENNet(
    in_features=3,
    hidden_features=128,
    hidden_layers=3,
    out_features=64,
    omega_0=30.0
)
output = net(coords)
```

---

### 2. `recurrent_parts.py` - 循环网络编码器

**核心类**:
- `GlobalEnvEncoder`: LSTM 编码器（处理 Kp/F10.7）
- `SpatialContextEncoder`: ConvLSTM 编码器（处理 TEC 地图）
- `ConvLSTMCell`: ConvLSTM 单元
- `ConvLSTM`: ConvLSTM 序列处理器

**使用示例**:
```python
from inr_modules.r_stmrf.recurrent_parts import GlobalEnvEncoder, SpatialContextEncoder

# 环境编码器
env_encoder = GlobalEnvEncoder(input_dim=2, hidden_dim=64, num_layers=2)
z_env = env_encoder(sw_seq)  # [Batch, Seq, 2] -> [Batch, 64]

# 空间编码器
spatial_encoder = SpatialContextEncoder(input_dim=1, hidden_dim=32)
F_tec = spatial_encoder(tec_seq)  # [Batch, Seq, 1, H, W] -> [Batch, 32, H, W]
```

---

### 3. `r_stmrf_model.py` - 主模型

**核心类**:
- `R_STMRF_Model`: 完整的 R-STMRF 模型

**模型架构**:
```
Ne = IRI_frozen + Decoder(h_spatial_mod, h_temporal_mod)

其中:
  h_spatial_mod = γ ⊙ h_spatial + β  (FiLM 调制)
  h_temporal_mod = h_temporal + β    (加性调制)
```

**使用示例**:
```python
from inr_modules.r_stmrf import R_STMRF_Model

model = R_STMRF_Model(
    iri_proxy=iri_proxy,
    lat_range=(-90, 90),
    lon_range=(-180, 180),
    alt_range=(120, 500),
    config=config
)

pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_map_seq)
```

---

### 4. `physics_losses_r_stmrf.py` - 物理损失

**核心函数**:
- `chapman_smoothness_loss()`: Chapman 垂直平滑损失
- `tec_gradient_alignment_loss_v2()`: TEC 梯度对齐损失
- `combined_physics_loss()`: 组合物理损失

**使用示例**:
```python
from inr_modules.r_stmrf.physics_losses_r_stmrf import combined_physics_loss

loss_physics, loss_dict = combined_physics_loss(
    pred_ne=pred_ne,
    coords=coords,
    target_tec_map=tec_map,
    w_chapman=0.1,
    w_tec_align=0.05
)
```

---

### 5. `sliding_dataset.py` - 数据处理

**核心类**:
- `SlidingWindowBatchProcessor`: 批次数据处理器

**使用示例**:
```python
from inr_modules.r_stmrf.sliding_dataset import SlidingWindowBatchProcessor

processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)
coords, target_ne, sw_seq, tec_map_seq, target_tec_map = processor.process_batch(batch_data)
```

---

### 6. `config_r_stmrf.py` - 配置文件

**核心函数**:
- `get_config_r_stmrf()`: 获取配置字典
- `print_config_r_stmrf()`: 打印配置
- `update_config_r_stmrf()`: 更新配置

**主要配置参数**:
```python
CONFIG_R_STMRF = {
    # 数据路径
    'fy_path': 'path/to/fy_data.npy',
    'tec_path': 'path/to/tec_map_data.npy',
    
    # 模型参数
    'seq_len': 6,
    'basis_dim': 64,
    'siren_hidden': 128,
    
    # 损失权重
    'w_chapman': 0.1,
    'w_tec_align': 0.05,
    
    # 训练参数
    'batch_size': 1024,
    'lr': 3e-4,
    'epochs': 50,
}
```

---

### 7. `train_r_stmrf.py` - 训练脚本

**核心函数**:
- `train_one_epoch()`: 训练一个 epoch
- `validate()`: 验证模型
- `train_r_stmrf()`: 完整训练流程

**使用示例**:
```python
from inr_modules.r_stmrf.train_r_stmrf import train_r_stmrf
from inr_modules.r_stmrf.config_r_stmrf import get_config_r_stmrf

config = get_config_r_stmrf()
model, train_losses, val_losses, *_ = train_r_stmrf(config)
```

---

### 8. `main_r_stmrf.py` - 主入口

**功能**:
- 完整的训练+评估+可视化流程
- 调用 `train_r_stmrf()` 执行训练
- 绘制损失曲线
- 保存模型和结果

**运行方式**:
```bash
python main_r_stmrf.py
```

---

## 🔧 安装步骤

### 步骤 1: 解压文件

```bash
# 假设当前在项目根目录
tar -xzf r_stmrf_modules.tar.gz
```

### 步骤 2: 验证文件结构

```bash
# 应该看到以下结构
project_root/
├── inr_modules/
│   ├── r_stmrf/
│   │   ├── __init__.py
│   │   ├── siren_layers.py
│   │   ├── recurrent_parts.py
│   │   ├── r_stmrf_model.py
│   │   ├── physics_losses_r_stmrf.py
│   │   ├── sliding_dataset.py
│   │   ├── config_r_stmrf.py
│   │   ├── train_r_stmrf.py
│   │   └── README_R_STMRF.md
│   └── data_managers/
│       └── tec_manager.py (确保包含 get_tec_map_sequence 方法)
├── main_r_stmrf.py
└── R_STMRF_IMPLEMENTATION_SUMMARY.md
```

### 步骤 3: 配置路径

编辑 `inr_modules/r_stmrf/config_r_stmrf.py`，修改数据路径：

```python
CONFIG_R_STMRF = {
    'fy_path': r'/your/path/to/fy_data.npy',
    'iri_proxy_path': r'/your/path/to/iri_proxy.pth',
    'sw_path': r'/your/path/to/kp_f107.txt',
    'tec_path': r'/your/path/to/tec_map_data.npy',
    'save_dir': './checkpoints_r_stmrf',
    # ...
}
```

### 步骤 4: 测试安装

```bash
# 测试模块导入
python -c "from inr_modules.r_stmrf import R_STMRF_Model; print('✓ 安装成功')"

# 运行单元测试
python -m inr_modules.r_stmrf.siren_layers
python -m inr_modules.r_stmrf.recurrent_parts
```

### 步骤 5: 开始训练

```bash
python main_r_stmrf.py
```

---

## ⚠️ 重要提示

### 依赖的现有模块

R-STMRF 依赖以下现有模块（应已存在于项目中）:

1. **`inr_modules/data_managers/`**
   - `FY_dataloader.py`: FY 数据加载器
   - `space_weather_manager.py`: 空间天气管理器
   - `tec_manager.py`: TEC 管理器（已修改，添加 `get_tec_map_sequence()`）
   - `irinc_neural_proxy.py`: IRI 神经代理

2. **标准库**
   - torch, numpy, pandas, matplotlib, tqdm

### TEC 数据格式要求

- **原始格式**: `(T, 71, 73)` numpy array
- **自动上采样**: → `(181, 361)`
- **纬度**: [-87.5, 87.5], 步长 2.5°
- **经度**: [-180, 180], 步长 5°

---

## 📞 技术支持

### 常见问题

**Q1: 显存不足怎么办？**
```python
# 在 config_r_stmrf.py 中调整:
'batch_size': 512,        # 减小批次
'tec_feat_dim': 16,       # 减少通道数
'siren_hidden': 64,       # 减小隐层
```

**Q2: 训练不收敛？**
- 检查学习率（建议 1e-4 ~ 5e-4）
- 启用梯度裁剪（`grad_clip=1.0`）
- 调整物理损失权重（`w_chapman` 不要太大）

**Q3: 如何可视化中间特征？**
```python
pred_ne, log_var, correction, extras = model(coords, sw_seq, tec_map_seq)

# 查看调制参数
gamma = extras['gamma']              # 空间缩放
beta_temporal = extras['beta_temporal']  # 时间偏移
F_tec = extras['F_tec']              # TEC 特征图
```

---

## 📚 文档链接

- **快速开始**: `inr_modules/r_stmrf/README_R_STMRF.md`
- **技术详解**: `R_STMRF_IMPLEMENTATION_SUMMARY.md`
- **文件清单**: `FILE_MANIFEST.md`
- **本指南**: `DOWNLOAD_GUIDE.md`

---

生成时间: 2026-01-28
版本: v1.0

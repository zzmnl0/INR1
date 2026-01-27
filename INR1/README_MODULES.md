# 物理引导的隐式神经辐射场 (Physics-Guided INR)

## 项目简介

本项目实现了基于物理约束的电离层电子密度三维重构系统，采用模块化架构设计，便于维护和升级。

## 目录结构

```
inr_modules/
├── __init__.py                    # 包初始化
├── config.py                      # 全局配置
├── train.py                       # 训练脚本
│
├── data_managers/                 # 数据管理模块
│   ├── __init__.py
│   ├── space_weather_manager.py   # 空间天气数据 (Kp/F10.7)
│   └── tec_manager.py             # TEC数据管理
│
├── models/                        # 模型组件
│   ├── __init__.py
│   ├── basic_layers.py            # 基础层（傅里叶编码）
│   ├── basis_networks.py          # 空间/时间基函数网络
│   ├── modulation_networks.py     # 调制网络（SW/VTEC）
│   └── inr_model.py               # 主INR模型
│
├── losses/                        # 损失函数
│   ├── __init__.py
│   └── physics_losses.py          # 物理约束损失
│
└── utils/                         # 工具函数
    ├── __init__.py
    ├── evaluation.py              # 评估工具
    └── visualization.py           # 可视化工具

main_inr.py                        # 主程序入口
```

## 快速开始

### 1. 基本使用

```python
# 运行完整流程（训练+评估+可视化）
python main_inr.py
```

### 2. 仅训练模型

```python
from inr_modules.config import get_config
from inr_modules.train import train_model

config = get_config()
model, train_losses, val_losses, *managers = train_model(config)
```

### 3. 自定义配置

```python
from inr_modules.config import get_config, update_config

# 修改配置
update_config(
    epochs=100,
    batch_size=4096,
    lr=1e-3
)

config = get_config()
```

## 模块说明

### 1. 配置模块 (`config.py`)

**功能**: 统一管理所有超参数

**主要参数**:
- `fy_path`: FY卫星数据路径
- `iri_proxy_path`: IRI神经代理路径
- `sw_path`: 空间天气数据路径
- `tec_path`: TEC数据路径
- `seq_len`: 历史窗口长度
- `batch_size`: 批次大小
- `lr`: 学习率
- `epochs`: 训练轮数

**常用函数**:
```python
get_config()           # 获取配置字典
update_config(**kw)    # 更新配置
print_config()         # 打印配置
```

### 2. 数据管理器模块 (`data_managers/`)

#### 2.1 空间天气管理器

**类**: `SpaceWeatherManager`

**功能**: 
- 加载并预处理Kp/F10.7数据
- 生成时序滑动窗口
- 支持零填充和掩码

**使用示例**:
```python
from inr_modules.data_managers import SpaceWeatherManager

sw_manager = SpaceWeatherManager(
    txt_path='path/to/sw_data.txt',
    start_date_str='2024-09-01 00:00:00',
    total_hours=720.0,
    seq_len=6,
    device='cuda'
)

# 获取历史序列
time_batch = torch.tensor([10.5, 20.3, 30.1])
sw_seq = sw_manager.get_drivers_sequence(time_batch)
# 返回: [Batch, Seq_Len, 2] (Kp, F10.7)
```

#### 2.2 TEC管理器

**类**: `TECDataManager`

**功能**:
- 加载并上采样TEC地图
- 提供时空插值
- 支持grid_sample

**使用示例**:
```python
from inr_modules.data_managers import TECDataManager

tec_manager = TECDataManager(
    tec_map_path='path/to/tec_data.npy',
    total_hours=720.0,
    seq_len=6,
    device='cuda'
)

# 获取TEC序列
lat = torch.tensor([30.0, -20.0])
lon = torch.tensor([120.0, -60.0])
time = torch.tensor([15.0, 25.0])

tec_seq = tec_manager.get_tec_sequence(lat, lon, time)
# 返回: [Batch, Seq_Len, 1]
```

### 3. 模型组件模块 (`models/`)

#### 3.1 基础层

**类**: `FourierFeatureEncoding`

**功能**: 傅里叶特征编码

```python
from inr_modules.models import FourierFeatureEncoding

encoder = FourierFeatureEncoding(input_dim=3, mapping_size=128, scale=1.0)
features = encoder(coords)  # [Batch, 3] -> [Batch, 256]
```

#### 3.2 主INR模型

**类**: `PhysicsGuidedINR`

**架构**: `Ne = IRI + Alpha * Correction`

**使用示例**:
```python
from inr_modules.models import PhysicsGuidedINR

model = PhysicsGuidedINR(
    iri_proxy=iri_proxy,
    lat_range=(-90.0, 90.0),
    lon_range=(-180.0, 180.0),
    alt_range=(120.0, 500.0),
    sw_manager=sw_manager,
    tec_manager=tec_manager,
    start_date_str='2024-09-01 00:00:00',
    config=config
)

# 前向传播
coords = torch.randn(32, 4)  # [Batch, 4] (Lat, Lon, Alt, Time)
sw_seq = torch.randn(32, 6, 2)  # [Batch, Seq, 2]
tec_seq = torch.randn(32, 6, 1)  # [Batch, Seq, 1]

ne_pred, log_var, correction, delta_w, gamma, beta = model(coords, sw_seq, tec_seq)
```

### 4. 损失函数模块 (`losses/`)

**主要损失函数**:

```python
from inr_modules.losses import (
    heteroscedastic_loss,           # 异方差损失
    iri_gradient_direction_loss,    # IRI梯度一致性
    tec_gradient_alignment_loss,    # TEC梯度对齐
    smoothness_loss_tv,             # 平滑约束
    get_obs_weight                  # 观测权重
)

# 使用示例
loss_obs = heteroscedastic_loss(pred, target, log_var, obs_weight)
loss_iri = iri_gradient_direction_loss(pred_ne, background_ne, coords)
loss_tec = tec_gradient_alignment_loss(pred_ne, coords, tec_manager)
loss_smooth = smoothness_loss_tv(pred_ne, coords)
```

### 5. 工具函数模块 (`utils/`)

#### 5.1 评估工具

```python
from inr_modules.utils import (
    evaluate_and_save_report,  # 生成评估报告
    plot_loss_curve,           # 绘制损失曲线
    evaluate_parity            # 绘制Parity图
)

# 生成评估报告
evaluate_and_save_report(
    model, train_loader, val_loader,
    sw_manager, tec_manager, device, save_dir
)

# 绘制损失曲线
plot_loss_curve(train_losses, val_losses, save_path)

# 绘制Parity图
evaluate_parity(
    model, val_loader, sw_manager, tec_manager,
    device, save_dir, config
)
```

#### 5.2 可视化工具

```python
from inr_modules.utils import plot_full_altitude_profile

# 绘制高度剖面
plot_full_altitude_profile(
    model, sw_manager, tec_manager, device,
    target_day=5, target_hour=12.0,
    save_dir='./results'
)
```

## 升级指南

### 升级单个模块

由于采用模块化设计，可以独立升级各个组件：

#### 1. 升级数据管理器

```python
# 在 data_managers/tec_manager.py 中修改
class TECDataManager:
    def get_tec_sequence(self, lat, lon, time_end):
        # 添加新功能
        # ... 新代码 ...
        pass
```

#### 2. 升级模型架构

```python
# 在 models/inr_model.py 中修改
class PhysicsGuidedINR(nn.Module):
    def __init__(self, ...):
        # 添加新组件
        self.new_component = NewNetwork(...)
```

#### 3. 升级损失函数

```python
# 在 losses/physics_losses.py 中添加
def new_physics_loss(pred, target, ...):
    # 实现新损失
    return loss
```

### 添加新功能

#### 示例：添加新的调制网络

1. 在 `models/modulation_networks.py` 添加新类:

```python
class NewModulator(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 初始化
    
    def forward(self, x):
        # 实现前向传播
        return output
```

2. 在 `models/inr_model.py` 中集成:

```python
class PhysicsGuidedINR(nn.Module):
    def __init__(self, ...):
        # ...
        self.new_modulator = NewModulator(...)
    
    def forward(self, ...):
        # 使用新模块
        new_output = self.new_modulator(...)
```

3. 更新 `models/__init__.py`:

```python
from .modulation_networks import NewModulator

__all__ = [..., 'NewModulator']
```

## 常见问题

### Q1: 如何修改训练超参数？

```python
from inr_modules.config import update_config

update_config(
    epochs=100,
    batch_size=4096,
    lr=1e-3,
    w_iri_dir=0.3  # 修改损失权重
)
```

### Q2: 如何加载已训练模型？

```python
import torch
from inr_modules.models import PhysicsGuidedINR
from inr_modules.config import get_config

config = get_config()
model = PhysicsGuidedINR(...)

# 加载权重
state_dict = torch.load('checkpoints/best_model_vtec_mod.pth')
model.load_state_dict(state_dict)
model.eval()
```

### Q3: 如何只运行推理？

```python
# 参考 main_inr.py 中的可视化部分
from inr_modules.utils import plot_full_altitude_profile

plot_full_altitude_profile(
    model, sw_manager, tec_manager, device,
    target_day=10, target_hour=15.0,
    save_dir='./inference_results'
)
```

## 依赖包

```
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
scipy>=1.7.0
```

## 版本历史

- v1.0.0 (2024): 初始模块化版本
  - 完整的模块化架构
  - 支持VTEC空间调制
  - 支持SW时间调制
  - 动态融合门控

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请联系项目维护者。

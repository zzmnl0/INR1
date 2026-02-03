# R-STMRF vs INR: 可视化脚本对比说明

## 文件信息
- **原文件**: `INR1_ISR_IRI_plot.py` (main分支)
- **新文件**: `R_STMRF_ISR_IRI_plot.py` (当前分支)

## 主要更改汇总

### 1. 导入模块更改

#### INR1 版本 (原)
```python
from INR1 import PhysicsGuidedINR, SpaceWeatherManager, TECDataManager, IRINeuralProxy
```

#### R-STMRF 版本 (新)
```python
from r_stmrf.r_stmrf_model import R_STMRF_Model
from r_stmrf.tec_gradient_bank import TecGradientBank
from r_stmrf.sliding_dataset import SlidingWindowBatchProcessor
from data_managers.space_weather_manager import SpaceWeatherManager
from data_managers.tec_data_manager import TECDataManager
from utils.iri_neural_proxy import IRINeuralProxy
```

**说明**:
- 替换 `PhysicsGuidedINR` → `R_STMRF_Model`
- 新增 `TecGradientBank` (离线预计算的TEC梯度库)
- 新增 `SlidingWindowBatchProcessor` (批次处理器)

---

### 2. 配置参数更改

#### 新增参数
```python
'gradient_bank_path': r'D:\IGS\VTEC\tec_gradient_bank.npy',  # TEC 梯度库路径

# R-STMRF 模型配置 (需要传递给模型初始化)
'basis_dim': 64,
'siren_hidden': 128,
'siren_layers': 3,
'omega_0': 30.0,
'env_hidden_dim': 64,
```

#### 权重路径调整
```python
# 原: 'model_weights': r"D:\code11\IRI01\IRI03\INR1-1\checkpoints_r_stmrf\best_r_stmrf_model.pth"
# 新: 'model_weights': r"./checkpoints_r_stmrf/best_r_stmrf_model.pth"
```

---

### 3. 模型初始化更改 (`initialize_model` 函数)

#### INR1 版本
```python
def initialize_model(config):
    sw_manager = SpaceWeatherManager(...)
    tec_manager = TECDataManager(...)
    iri_proxy = IRINeuralProxy(...)

    model = PhysicsGuidedINR(
        iri_proxy=iri_proxy,
        lat_range=config['lat_range'],
        lon_range=config['lon_range'],
        alt_range=config['alt_range'],
        sw_manager=sw_manager,
        tec_manager=tec_manager,
        start_date_str=config['start_date_str']
    ).to(device)

    return model, sw_manager, tec_manager, device
```

#### R-STMRF 版本
```python
def initialize_model(config):
    sw_manager = SpaceWeatherManager(...)
    tec_manager = TECDataManager(...)

    # 新增: TEC 梯度库
    gradient_bank = TecGradientBank(
        gradient_bank_path=config['gradient_bank_path'],
        total_hours=config['total_hours'],
        device=device
    )

    iri_proxy = IRINeuralProxy(...)

    # 模型初始化需要 config 参数
    model = R_STMRF_Model(
        iri_proxy=iri_proxy,
        lat_range=config['lat_range'],
        lon_range=config['lon_range'],
        alt_range=config['alt_range'],
        sw_manager=sw_manager,
        tec_manager=tec_manager,
        start_date_str=config['start_date_str'],
        config=config  # 新增
    ).to(device)

    # 新增: 批次处理器
    batch_processor = SlidingWindowBatchProcessor(sw_manager, tec_manager, device)

    return model, sw_manager, tec_manager, gradient_bank, batch_processor, device
```

**关键变化**:
1. 新增 `TecGradientBank` 初始化（离线梯度查询）
2. 模型初始化增加 `config` 参数（传递 SIREN 架构参数）
3. 新增 `SlidingWindowBatchProcessor` 实例
4. 返回值增加 `gradient_bank` 和 `batch_processor`

---

### 4. 推理函数更改 (`run_*_iri_profile`)

#### 函数签名更改
```python
# 原: run_inr_iri_profile(model, sw_manager, tec_manager, config, device)
# 新: run_r_stmrf_iri_profile(model, batch_processor, gradient_bank, config, device)
```

#### SW 序列获取方式

**INR1 版本**:
```python
# 使用批次的平均时间查询
mean_time = batch_times.mean().unsqueeze(0)
sw_seq_single = sw_manager.get_drivers_sequence(mean_time)
sw_seq = sw_seq_single.expand(batch_n, -1, -1)
```

**R-STMRF 版本**:
```python
# 使用 batch_processor 的 sw_manager，支持点级查询
sw_seq = batch_processor.sw_manager.get_drivers_sequence(batch_times)  # [Batch, Seq, 2]
```

**说明**: R-STMRF 直接使用 `batch_times` 向量查询，每个点获取独立的 SW 序列

#### TEC 处理方式

**INR1 版本**:
```python
# 在线查询 TEC 地图序列
tec_seq = tec_manager.get_tec_sequence(
    batch_coords[:, 0],
    batch_coords[:, 1],
    batch_coords[:, 3]
)
```

**R-STMRF 版本**:
```python
# 从离线预计算的梯度库查询
tec_grad_direction = gradient_bank.get_interpolated_gradient(batch_times)  # [Batch, 2, H, W]
```

**说明**: R-STMRF v2.0+ 不使用在线 TEC 加载，改用离线预计算的梯度库

#### 模型调用

**INR1 版本** (6个返回值):
```python
output, _, _, _, _, _ = model(batch_coords, sw_seq, tec_seq)
inr_pred = output.squeeze()
```

**R-STMRF 版本** (4个返回值):
```python
r_stmrf_pred, log_var, correction, extras = model(batch_coords, sw_seq, tec_grad_direction)
r_stmrf_pred = r_stmrf_pred.squeeze()
log_var = log_var.squeeze()
```

**说明**:
- R-STMRF 返回 4 个值：`(output, log_var, correction, extras)`
- 新增 `log_var` 用于不确定性估计（异方差损失）

#### 返回值更改

**INR1 版本**:
```python
return timestamps, altitudes, inr_ne_2d, iri_ne_2d
```

**R-STMRF 版本**:
```python
return timestamps, altitudes, r_stmrf_ne_2d, iri_ne_2d, log_var_ne_2d
```

**说明**: 新增 `log_var_ne_2d` 用于不确定性可视化

---

### 5. 可视化更改

#### 5-Panel 对比图

**标题更改**:
```python
# 原: '(b) Physics-Guided INR Prediction'
# 新: '(b) R-STMRF Model Prediction'

# 原: '(d) Difference: INR − ISR'
# 新: '(d) Difference: R-STMRF − ISR'

# 原: 'INR vs IRI vs ISR Comparison'
# 新: 'R-STMRF vs IRI vs ISR Comparison'
```

**输出文件名**:
```python
# 原: 'inr_iri_isr_5panel_comparison.png'
# 新: 'r_stmrf_iri_isr_5panel_comparison.png'
```

#### 新增: 不确定性可视化 (独立画布)

```python
def plot_uncertainty(r_stmrf_timestamps, r_stmrf_altitudes, log_var_ne, config):
    """
    Create separate figure for R-STMRF prediction uncertainty (log_var).
    独立画布展示 R-STMRF 的预测不确定性（对数方差）
    """
    # 使用 'viridis' 色图显示 log_var
    # 输出文件: 'r_stmrf_uncertainty.png'
    # 统计: log_var 范围、均值、标准差，以及对应的 σ (标准差)
```

**特性**:
- 独立的 1x1 子图布局
- 显示 log(σ²) (对数方差)
- 提供统计信息（范围、均值、标准差）
- 转换为 σ (标准差) 以便理解

---

### 6. 主函数更改

#### INR1 版本
```python
def main():
    # 1. 初始化模型
    model, sw_manager, tec_manager, device = initialize_model(CONFIG)

    # 2. 推理
    inr_timestamps, inr_altitudes, inr_ne, iri_ne = run_inr_iri_profile(...)

    # 3. 读取 ISR
    isr_data = read_isr_data(...)

    # 4. 插值
    inr_on_isr = interpolate_to_isr_grid(...)
    iri_on_isr = interpolate_to_isr_grid(...)

    # 5. 绘制 5-Panel 对比
    plot_5panel_comparison(...)
```

#### R-STMRF 版本
```python
def main():
    # 1. 初始化模型 (返回值增加 gradient_bank, batch_processor)
    model, sw_manager, tec_manager, gradient_bank, batch_processor, device = initialize_model(CONFIG)

    # 2. 推理 (返回值增加 log_var_ne)
    r_stmrf_timestamps, r_stmrf_altitudes, r_stmrf_ne, iri_ne, log_var_ne = run_r_stmrf_iri_profile(...)

    # 3. 读取 ISR
    isr_data = read_isr_data(...)

    # 4. 插值
    r_stmrf_on_isr = interpolate_to_isr_grid(...)
    iri_on_isr = interpolate_to_isr_grid(...)

    # 5. 绘制 5-Panel 对比
    plot_5panel_comparison(...)

    # 6. 绘制不确定性 (新增)
    plot_uncertainty(r_stmrf_timestamps, r_stmrf_altitudes, log_var_ne, CONFIG)
```

---

## 架构差异总结

| 特性 | INR1 | R-STMRF v2.0+ |
|------|------|---------------|
| **TEC 处理** | 在线查询 TEC 地图序列 | 离线预计算梯度库 + 时间插值 |
| **模型输出** | 6个返回值 | 4个返回值 (output, log_var, correction, extras) |
| **SW 序列** | 批次平均时间 + 广播 | 点级查询 (每点独立序列) |
| **不确定性** | 无 | 异方差损失 (log_var) |
| **内存占用** | 较高 (在线 TEC 加载) | 极低 (Memory-Mapped 梯度库) |
| **推理速度** | 较慢 | 快速 (无 ConvLSTM 在线计算) |

---

## 输出文件

### 原 INR1 版本
- `inr_iri_isr_5panel_comparison.png` (5个子图)

### 新 R-STMRF 版本
- `r_stmrf_iri_isr_5panel_comparison.png` (5个子图)
- `r_stmrf_uncertainty.png` (独立画布，不确定性可视化)

---

## 使用说明

### 前置要求
1. 确保已预计算 `tec_gradient_bank.npy` (使用 `precompute_tec_gradient_bank.py`)
2. 确保已训练 R-STMRF 模型 (`best_r_stmrf_model.pth`)
3. 确保 IRI 神经代理已训练 (`iri_september_full_proxy.pth`)

### 运行脚本
```bash
python R_STMRF_ISR_IRI_plot.py
```

### 输出
- 终端输出：模型加载进度、推理统计、评估指标
- 图片1：5-Panel 对比图 (R-STMRF vs IRI vs ISR)
- 图片2：不确定性可视化 (log_var)

---

## 技术要点

### R-STMRF v2.0+ 架构优势
1. **离线梯度预计算**: 消除 ConvLSTM 在线计算开销，大幅降低内存和推理时间
2. **Memory-Mapped 加载**: TEC 梯度库使用 mmap，RAM 占用 < 10 MB
3. **余弦平方插值**: 时间维度保证 C1 连续性，无高频泄露
4. **异方差不确定性**: 学习预测方差，提供置信度估计

### 关键 API 变化
```python
# 模型 forward()
# 原: output, _, _, _, _, _ = model(coords, sw_seq, tec_seq)
# 新: output, log_var, correction, extras = model(coords, sw_seq, tec_grad_direction)

# TEC 查询
# 原: tec_seq = tec_manager.get_tec_sequence(lat, lon, time)
# 新: tec_grad_direction = gradient_bank.get_interpolated_gradient(time)

# SW 查询
# 原: sw_seq = sw_manager.get_drivers_sequence(mean_time).expand(batch_n, -1, -1)
# 新: sw_seq = batch_processor.sw_manager.get_drivers_sequence(batch_times)
```

---

## 致谢
- 基于原 `INR1_ISR_IRI_plot.py` 架构
- 适配 R-STMRF v2.0+ 模型 API
- 新增不确定性可视化功能

---

**文档版本**: v1.0
**创建日期**: 2026-02-03
**作者**: Claude AI (R-STMRF 团队)

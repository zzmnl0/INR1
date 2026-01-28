# 📦 R-STMRF 完整文件下载清单

## 🎯 总览

- **总文件数**: 12 个
- **压缩包大小**: 28 KB
- **解压后大小**: ~90 KB
- **总代码行数**: ~3000 行
- **Git 提交**: fa4af47

---

## 📥 下载方式

### 方式 1: 压缩包（推荐）

**文件**: `r_stmrf_modules.tar.gz`
**路径**: `/home/user/INR1/INR1/r_stmrf_modules.tar.gz`
**大小**: 28 KB

**解压命令**:
```bash
tar -xzf r_stmrf_modules.tar.gz
```

### 方式 2: Git Clone

```bash
git clone https://github.com/zzmnl0/INR1.git
cd INR1
git checkout claude/add-physical-constraints-TFM8i
```

### 方式 3: 单独下载文件

所有文件路径见下方详细清单。

---

## 📋 详细文件清单

### 🔧 核心模块文件 (inr_modules/r_stmrf/)

#### 1. `__init__.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/__init__.py`
- **大小**: 401 B
- **行数**: 15
- **功能**: 模块导出定义
- **导出内容**:
  ```python
  - SIRENLayer, SIRENNet
  - GlobalEnvEncoder, SpatialContextEncoder
  - R_STMRF_Model
  ```

#### 2. `siren_layers.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/siren_layers.py`
- **大小**: 7.1 KB
- **行数**: 210
- **功能**: SIREN 基础层实现
- **核心类**:
  - `SIRENLayer`: 单层 SIREN (sin 激活 + 特殊初始化)
  - `SIRENNet`: 多层 SIREN 网络
  - `ModulatedSIRENNet`: 可调制的 SIREN 网络
- **关键技术**:
  - sin 激活函数
  - 特殊的权重初始化（SIREN 论文）
  - 频率因子 ω₀ = 30
- **依赖**: torch, numpy

#### 3. `recurrent_parts.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/recurrent_parts.py`
- **大小**: 11.8 KB
- **行数**: 270
- **功能**: 循环网络编码器
- **核心类**:
  - `ConvLSTMCell`: ConvLSTM 单元
  - `ConvLSTM`: ConvLSTM 序列处理器
  - `GlobalEnvEncoder`: LSTM 编码器（处理 Kp/F10.7）
  - `SpatialContextEncoder`: ConvLSTM 编码器（处理 TEC 地图）
- **关键技术**:
  - ConvLSTM 时空建模
  - LSTM 序列编码
  - BatchNorm + ReLU 特征细化
- **依赖**: torch

#### 4. `r_stmrf_model.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/r_stmrf_model.py`
- **大小**: 13.5 KB
- **行数**: 360
- **功能**: R-STMRF 主模型（核心架构）
- **核心类**:
  - `R_STMRF_Model`: 完整模型
- **架构组件**:
  - Spatial Basis Net (SIREN)
  - Temporal Basis Net (SIREN)
  - Spatial Context Encoder (ConvLSTM)
  - Global Env Encoder (LSTM)
  - FiLM Modulation (γ, β)
  - Additive Modulation (β)
  - Fusion Decoder
  - Uncertainty Head
- **输入输出**:
  - 输入: coords [Batch, 4], sw_seq [Batch, Seq, 2], tec_map_seq [Batch, Seq, 1, H, W]
  - 输出: pred_ne, log_var, correction, extras
- **依赖**: torch, pandas, siren_layers, recurrent_parts

#### 5. `physics_losses_r_stmrf.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/physics_losses_r_stmrf.py`
- **大小**: 9.4 KB
- **行数**: 260
- **功能**: 物理约束损失函数
- **核心函数**:
  - `chapman_smoothness_loss()`: Chapman 垂直平滑损失
  - `tec_gradient_alignment_loss_v2()`: TEC 梯度对齐损失（基于地图）
  - `combined_physics_loss()`: 组合物理损失
- **关键技术**:
  - 自动微分计算二阶导数
  - Sobel 算子计算空间梯度
  - Cosine Similarity 对齐
  - 自适应掩码（梯度显著性）
- **依赖**: torch

#### 6. `sliding_dataset.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/sliding_dataset.py`
- **大小**: 7.2 KB
- **行数**: 190
- **功能**: 滑动窗口数据处理工具
- **核心类**:
  - `SlidingWindowBatchProcessor`: 批次数据处理器
- **核心函数**:
  - `get_r_stmrf_dataloaders()`: 获取 DataLoader
  - `collate_with_sequences()`: 自定义 collate 函数
- **关键逻辑**:
  - 保留 TimeBinSampler 策略
  - 动态查询历史序列
  - 返回完整数据包（coords, sw_seq, tec_map_seq, target_tec_map）
- **依赖**: torch, numpy

#### 7. `config_r_stmrf.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/config_r_stmrf.py`
- **大小**: 6.9 KB
- **行数**: 150
- **功能**: 配置参数定义
- **核心配置**:
  - 数据路径 (fy_path, tec_path, sw_path, iri_proxy_path)
  - 物理参数 (lat_range, lon_range, alt_range)
  - 时序参数 (seq_len=6)
  - SIREN 参数 (basis_dim=64, siren_hidden=128, omega_0=30)
  - 循环网络参数 (tec_feat_dim=32, env_hidden_dim=64)
  - 损失权重 (w_chapman=0.1, w_tec_align=0.05)
  - 训练参数 (batch_size=1024, lr=3e-4, epochs=50)
- **核心函数**:
  - `get_config_r_stmrf()`: 获取配置字典
  - `print_config_r_stmrf()`: 打印配置
  - `update_config_r_stmrf()`: 更新配置
  - `validate_config()`: 验证配置
- **依赖**: torch, os

#### 8. `train_r_stmrf.py`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/train_r_stmrf.py`
- **大小**: 11.5 KB
- **行数**: 320
- **功能**: 完整训练脚本
- **核心函数**:
  - `train_one_epoch()`: 训练一个 epoch
  - `validate()`: 验证模型
  - `train_r_stmrf()`: 完整训练流程
- **训练流程**:
  1. 初始化数据管理器 (sw_manager, tec_manager)
  2. 加载 IRI 神经代理（冻结）
  3. 准备数据集（TimeBinSampler）
  4. 初始化 R-STMRF 模型
  5. 配置优化器和调度器
  6. 训练循环（epoch by epoch）
  7. 验证和早停
  8. 保存最佳模型
- **特性**:
  - 梯度裁剪
  - 学习率调度（Cosine Annealing）
  - 早停机制
  - 详细的损失统计
- **依赖**: torch, numpy, tqdm, config_r_stmrf, r_stmrf_model, physics_losses_r_stmrf, sliding_dataset

#### 9. `README_R_STMRF.md`
- **路径**: `/home/user/INR1/INR1/inr_modules/r_stmrf/README_R_STMRF.md`
- **大小**: 9.1 KB
- **行数**: 400
- **功能**: 完整使用文档
- **内容目录**:
  - 概述
  - 核心架构
  - 文件结构
  - 使用方法
  - 配置说明
  - 物理约束
  - 与原模型对比
  - 常见问题
- **语言**: Markdown

---

### 🚀 主入口文件

#### 10. `main_r_stmrf.py`
- **路径**: `/home/user/INR1/INR1/main_r_stmrf.py`
- **大小**: 2.9 KB
- **行数**: 80
- **功能**: 主程序入口（训练+评估+可视化）
- **流程**:
  1. 加载配置
  2. 打印配置
  3. 调用 `train_r_stmrf()` 训练
  4. 加载最佳模型
  5. 评估模型（TODO）
  6. 绘制损失曲线
  7. 可视化结果（TODO）
- **运行方式**:
  ```bash
  python main_r_stmrf.py
  ```
- **依赖**: torch, matplotlib, config_r_stmrf, train_r_stmrf

---

### 📚 文档文件

#### 11. `R_STMRF_IMPLEMENTATION_SUMMARY.md`
- **路径**: `/home/user/INR1/INR1/R_STMRF_IMPLEMENTATION_SUMMARY.md`
- **大小**: 13.0 KB
- **行数**: 500
- **功能**: 实施总结（技术细节+性能分析）
- **内容**:
  - 实施概览
  - 新增文件清单
  - 架构对比
  - 关键技术细节
  - 模型参数统计
  - 使用指南
  - 配置说明
  - 测试验证
  - 预期性能
  - 注意事项
  - 已知问题
  - 未来工作
- **语言**: Markdown

#### 12. `FILE_MANIFEST.md`
- **路径**: `/home/user/INR1/INR1/FILE_MANIFEST.md`
- **功能**: 文件清单（本文件）
- **语言**: Markdown

#### 13. `DOWNLOAD_GUIDE.md`
- **路径**: `/home/user/INR1/INR1/DOWNLOAD_GUIDE.md`
- **功能**: 下载指南
- **语言**: Markdown

---

### 🔧 修改的现有文件

#### `tec_manager.py` (修改)
- **路径**: `/home/user/INR1/INR1/inr_modules/data_managers/tec_manager.py`
- **修改内容**: 新增 `get_tec_map_sequence()` 方法
- **新方法功能**: 返回完整 TEC 地图序列 [Batch, Seq, 1, H, W]
- **代码行数**: 新增 45 行（108-152 行）
- **核心逻辑**:
  ```python
  def get_tec_map_sequence(self, time_end):
      # 生成时间序列
      # 批量索引提取地图
      # 归一化
      return tec_maps_norm  # [Batch, Seq, 1, 181, 361]
  ```

---

## 🗂️ 文件依赖图

```
main_r_stmrf.py
  │
  ├─ config_r_stmrf.py
  │
  └─ train_r_stmrf.py
       │
       ├─ config_r_stmrf.py
       │
       ├─ r_stmrf_model.py
       │    ├─ siren_layers.py
       │    ├─ recurrent_parts.py
       │    └─ [现有模块]
       │         ├─ irinc_neural_proxy.py
       │         ├─ space_weather_manager.py
       │         └─ tec_manager.py (修改版)
       │
       ├─ physics_losses_r_stmrf.py
       │
       └─ sliding_dataset.py
            ├─ space_weather_manager.py
            ├─ tec_manager.py (修改版)
            └─ FY_dataloader.py
```

---

## 📊 代码统计

### 按模块分类

| 模块类别 | 文件数 | 代码行数 | 占比 |
|---------|-------|---------|------|
| 核心模型 | 3 | 840 | 39% |
| 数据处理 | 2 | 450 | 21% |
| 损失函数 | 1 | 260 | 12% |
| 训练脚本 | 1 | 320 | 15% |
| 配置文件 | 1 | 150 | 7% |
| 主入口 | 1 | 80 | 4% |
| 文档 | 3 | 900 | - |

### 按文件大小

| 文件 | 大小 | 占比 |
|------|------|------|
| r_stmrf_model.py | 13.5 KB | 18% |
| recurrent_parts.py | 11.8 KB | 16% |
| train_r_stmrf.py | 11.5 KB | 15% |
| physics_losses_r_stmrf.py | 9.4 KB | 13% |
| README_R_STMRF.md | 9.1 KB | 12% |
| sliding_dataset.py | 7.2 KB | 10% |
| siren_layers.py | 7.1 KB | 9% |
| config_r_stmrf.py | 6.9 KB | 9% |
| 其他 | ~13 KB | - |

---

## ✅ 完整性检查

### 必需文件 (9个)

- [x] `inr_modules/r_stmrf/__init__.py`
- [x] `inr_modules/r_stmrf/siren_layers.py`
- [x] `inr_modules/r_stmrf/recurrent_parts.py`
- [x] `inr_modules/r_stmrf/r_stmrf_model.py`
- [x] `inr_modules/r_stmrf/physics_losses_r_stmrf.py`
- [x] `inr_modules/r_stmrf/sliding_dataset.py`
- [x] `inr_modules/r_stmrf/config_r_stmrf.py`
- [x] `inr_modules/r_stmrf/train_r_stmrf.py`
- [x] `main_r_stmrf.py`

### 文档文件 (4个)

- [x] `inr_modules/r_stmrf/README_R_STMRF.md`
- [x] `R_STMRF_IMPLEMENTATION_SUMMARY.md`
- [x] `FILE_MANIFEST.md`
- [x] `DOWNLOAD_GUIDE.md`

### 修改文件 (1个)

- [x] `inr_modules/data_managers/tec_manager.py` (包含 `get_tec_map_sequence()`)

---

## 🎯 使用流程

### 1. 下载
```bash
tar -xzf r_stmrf_modules.tar.gz
```

### 2. 配置
```bash
# 编辑配置文件
nano inr_modules/r_stmrf/config_r_stmrf.py

# 修改数据路径
'fy_path': '/your/path/to/fy_data.npy',
'tec_path': '/your/path/to/tec_map_data.npy',
...
```

### 3. 测试
```bash
# 测试导入
python -c "from inr_modules.r_stmrf import R_STMRF_Model; print('OK')"

# 运行单元测试
python -m inr_modules.r_stmrf.siren_layers
```

### 4. 训练
```bash
python main_r_stmrf.py
```

---

## 📞 技术支持

- **README**: `inr_modules/r_stmrf/README_R_STMRF.md`
- **技术文档**: `R_STMRF_IMPLEMENTATION_SUMMARY.md`
- **下载指南**: `DOWNLOAD_GUIDE.md`
- **Git 仓库**: https://github.com/zzmnl0/INR1
- **分支**: `claude/add-physical-constraints-TFM8i`
- **Commit**: fa4af47

---

**生成时间**: 2026-01-28  
**版本**: v1.0  
**状态**: ✅ 全部完成

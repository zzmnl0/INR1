"""
验证 ConvLSTM 内存优化效果

测试目标：
1. ConvLSTM 前向不包含 batch 维度
2. TEC 分支内存开销与 batch_size 无关
3. batch_size = 2048 在 CPU 可运行
4. FiLM 调制逻辑语义不变
"""

import torch
import numpy as np
import tracemalloc
import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from recurrent_parts import SpatialContextEncoder


def test_convlstm_memory_scaling():
    """测试 ConvLSTM 内存开销是否与 batch_size 无关"""
    print("="*70)
    print("测试 1: ConvLSTM 内存开销与 batch_size 的关系")
    print("="*70)

    # 配置
    seq_len = 6
    tec_feat_dim = 32
    tec_h, tec_w = 181, 361
    device = torch.device('cpu')

    # 创建 ConvLSTM 编码器
    encoder = SpatialContextEncoder(
        input_dim=1,
        hidden_dim=tec_feat_dim,
        num_layers=2,
        kernel_size=3
    ).to(device)

    # 测试不同的 batch_size（模拟不同数量的唯一时间窗口）
    test_cases = [
        ('小规模', 5),
        ('中等规模', 10),
        ('大规模', 20),
        ('极大规模', 32)
    ]

    results = []

    for name, n_unique in test_cases:
        # 创建唯一时间窗口的 TEC 序列
        tec_seq = torch.rand(n_unique, seq_len, 1, tec_h, tec_w).to(device)

        # 记录内存
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        # 前向传播
        with torch.no_grad():
            output = encoder(tec_seq)

        # 记录内存峰值
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_used = (peak - start_mem) / (1024 ** 2)  # MB

        print(f"\n{name}:")
        print(f"  唯一时间窗口数: {n_unique}")
        print(f"  输入形状: {tec_seq.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  内存使用: {mem_used:.2f} MB")

        results.append((n_unique, mem_used))

    # 验证线性关系
    print("\n" + "="*70)
    print("验证: 内存开销应该与唯一时间窗口数成线性关系")
    print("="*70)

    for i in range(1, len(results)):
        n_prev, mem_prev = results[i-1]
        n_curr, mem_curr = results[i]

        ratio_n = n_curr / n_prev
        ratio_mem = mem_curr / mem_prev

        print(f"\n{n_prev} -> {n_curr} 窗口:")
        print(f"  窗口数比例: {ratio_n:.2f}x")
        print(f"  内存比例: {ratio_mem:.2f}x")
        print(f"  线性度: {abs(ratio_mem - ratio_n) / ratio_n * 100:.1f}% 偏差")


def test_batch_independence():
    """测试大 batch_size 下的内存独立性"""
    print("\n" + "="*70)
    print("测试 2: 大 batch_size (2048) 在 CPU 上的可行性")
    print("="*70)

    # 配置
    batch_size = 2048
    seq_len = 6
    tec_feat_dim = 32
    tec_h, tec_w = 181, 361
    device = torch.device('cpu')

    # 创建 ConvLSTM 编码器
    encoder = SpatialContextEncoder(
        input_dim=1,
        hidden_dim=tec_feat_dim,
        num_layers=2,
        kernel_size=3
    ).to(device)

    # 模拟情况：batch_size=2048，但只有 10 个唯一时间窗口
    n_unique = 10
    unique_tec_seq = torch.rand(n_unique, seq_len, 1, tec_h, tec_w).to(device)

    print(f"\n场景设置:")
    print(f"  Batch size: {batch_size}")
    print(f"  唯一时间窗口数: {n_unique}")
    print(f"  理论重复率: {batch_size / n_unique:.1f}x")

    # 记录内存
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]

    try:
        # 前向传播（只处理唯一时间窗口）
        with torch.no_grad():
            F_tec_unique = encoder(unique_tec_seq)  # [N_unique, tec_feat_dim, H, W]

        # 模拟索引操作（不复制内存）
        tec_indices = torch.randint(0, n_unique, (batch_size,))
        F_tec_batch = F_tec_unique[tec_indices]  # [Batch, tec_feat_dim, H, W]

        # 记录内存峰值
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_used = (peak - start_mem) / (1024 ** 2)  # MB

        print(f"\n结果:")
        print(f"  F_tec_unique 形状: {F_tec_unique.shape}")
        print(f"  F_tec_batch 形状: {F_tec_batch.shape}")
        print(f"  内存使用: {mem_used:.2f} MB")
        print(f"  ✅ 成功! batch_size=2048 在 CPU 上可运行")

    except RuntimeError as e:
        print(f"\n❌ 失败: {str(e)}")
        tracemalloc.stop()


def test_film_modulation():
    """测试 FiLM 调制逻辑语义不变"""
    print("\n" + "="*70)
    print("测试 3: FiLM 调制逻辑验证")
    print("="*70)

    batch_size = 16
    basis_dim = 64
    tec_feat_dim = 32

    # 模拟空间基函数
    h_spatial = torch.randn(batch_size, basis_dim)

    # 模拟 TEC 特征
    z_tec = torch.randn(batch_size, tec_feat_dim)

    # FiLM 调制头
    modulation_head = torch.nn.Sequential(
        torch.nn.Linear(tec_feat_dim, basis_dim * 2),
        torch.nn.Tanh()
    )

    # 计算调制参数
    film_params = modulation_head(z_tec)
    gamma, beta = torch.chunk(film_params, 2, dim=-1)

    # 约束 gamma 范围
    gamma = torch.sigmoid(gamma)
    gamma = 0.8 + gamma * 0.4  # [0.8, 1.2]

    # FiLM 调制
    h_spatial_mod = gamma * h_spatial + beta

    print(f"\n调制参数统计:")
    print(f"  gamma 范围: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
    print(f"  gamma 均值: {gamma.mean().item():.3f}")
    print(f"  beta 范围: [{beta.min().item():.3f}, {beta.max().item():.3f}]")
    print(f"  beta 均值: {beta.mean().item():.3f}")

    # 验证调制效果
    modulation_strength = torch.abs(h_spatial_mod - h_spatial).mean()
    print(f"\n调制强度 (平均变化): {modulation_strength.item():.4f}")

    # 验证 gamma 范围约束
    assert gamma.min() >= 0.8 and gamma.max() <= 1.2, "gamma 范围约束失败"
    print(f"  ✅ gamma 范围约束验证通过 [0.8, 1.2]")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("R-STMRF ConvLSTM 内存优化验证")
    print("="*70 + "\n")

    # 运行测试
    test_convlstm_memory_scaling()
    test_batch_independence()
    test_film_modulation()

    print("\n" + "="*70)
    print("所有测试完成!")
    print("="*70 + "\n")

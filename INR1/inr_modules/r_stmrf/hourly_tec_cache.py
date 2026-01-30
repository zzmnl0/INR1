"""
小时级TEC上下文缓存管理器

职责:
- 管理ConvLSTM在整点小时的计算结果
- 提供LRU缓存机制，避免重复计算
- 支持按需计算和批量预计算

设计原则:
- ConvLSTM只在1h整点运行
- 输出低频、平滑的空间调制特征
- 不参与分钟级建模
"""

import torch
from collections import OrderedDict


class HourlyTECContextCache:
    """
    小时级TEC上下文特征缓存

    使用LRU策略管理ConvLSTM的计算结果
    """

    def __init__(self, convlstm_encoder, tec_gradient_head, max_cache_size=100, device='cpu'):
        """
        Args:
            convlstm_encoder: ConvLSTM编码器实例
            tec_gradient_head: TEC梯度方向提取头
            max_cache_size: 最大缓存小时数（默认100小时）
            device: 计算设备
        """
        self.convlstm_encoder = convlstm_encoder
        self.tec_gradient_head = tec_gradient_head
        self.max_cache_size = max_cache_size
        self.device = device

        # LRU缓存：OrderedDict保持插入顺序
        # key: hour (int), value: {'F_tec': tensor, 'grad_direction': tensor}
        # 注意：训练模式下禁用缓存，只在eval模式下使用（避免梯度图问题）
        self.cache = OrderedDict()

        # 统计信息
        self.hits = 0
        self.misses = 0

        # 缓存模式控制
        self.training_mode = True  # 跟踪是否在训练模式

    def get(self, hour_indices, tec_manager):
        """
        获取指定小时的TEC特征

        Args:
            hour_indices: [N_hours] 整点小时索引列表（去重后）
            tec_manager: TEC数据管理器

        Returns:
            F_tec_dict: {hour: [1, tec_feat_dim, H, W]}
            grad_direction_dict: {hour: [1, 2, H, W]}
        """
        # 训练模式下跳过缓存，直接计算（避免梯度图断裂）
        if self.convlstm_encoder.training:
            computed_F_tec, computed_grad = self._compute_batch(
                hour_indices, tec_manager
            )
            F_tec_dict = {int(hour): computed_F_tec[i:i+1] for i, hour in enumerate(hour_indices)}
            grad_direction_dict = {int(hour): computed_grad[i:i+1] for i, hour in enumerate(hour_indices)}
            return F_tec_dict, grad_direction_dict

        # 评估模式下使用缓存
        F_tec_dict = {}
        grad_direction_dict = {}

        hours_to_compute = []

        # 检查缓存
        for hour in hour_indices:
            hour_int = int(hour)

            if hour_int in self.cache:
                # 缓存命中
                self.hits += 1
                cached_data = self.cache[hour_int]

                # 移到末尾（LRU更新）
                self.cache.move_to_end(hour_int)

                F_tec_dict[hour_int] = cached_data['F_tec']
                grad_direction_dict[hour_int] = cached_data['grad_direction']
            else:
                # 缓存未命中
                self.misses += 1
                hours_to_compute.append(hour_int)

        # 批量计算未命中的小时
        if len(hours_to_compute) > 0:
            computed_F_tec, computed_grad = self._compute_batch(
                hours_to_compute, tec_manager
            )

            for i, hour in enumerate(hours_to_compute):
                F_tec_dict[hour] = computed_F_tec[i:i+1]
                grad_direction_dict[hour] = computed_grad[i:i+1]

                # 添加到缓存（eval模式下）
                self._add_to_cache(hour, computed_F_tec[i:i+1], computed_grad[i:i+1])

        return F_tec_dict, grad_direction_dict

    def _compute_batch(self, hour_indices, tec_manager):
        """
        批量计算多个小时的TEC特征

        Args:
            hour_indices: 整点小时索引列表
            tec_manager: TEC数据管理器

        Returns:
            F_tec: [N_hours, tec_feat_dim, H, W]
            grad_direction: [N_hours, 2, H, W]
        """
        # 获取TEC地图序列
        tec_map_seq = tec_manager.get_tec_map_sequence_by_hours(
            hour_indices
        )  # [N_hours, Seq, 1, H, W]

        # ConvLSTM编码（保留梯度以支持训练）
        F_tec = self.convlstm_encoder(tec_map_seq)  # [N_hours, tec_feat_dim, H, W]

        # 提取梯度方向
        grad_direction = self.tec_gradient_head(F_tec)  # [N_hours, 2, H, W]

        return F_tec, grad_direction

    def _add_to_cache(self, hour, F_tec, grad_direction):
        """
        添加到缓存，执行LRU淘汰（仅在eval模式下调用）

        Args:
            hour: 小时索引
            F_tec: [1, tec_feat_dim, H, W]
            grad_direction: [1, 2, H, W]
        """
        # 检查缓存大小
        if len(self.cache) >= self.max_cache_size:
            # LRU淘汰：移除最旧的条目（第一个）
            oldest_hour = next(iter(self.cache))
            del self.cache[oldest_hour]

        # 添加新条目（使用detach避免保留梯度图）
        self.cache[hour] = {
            'F_tec': F_tec.detach().clone(),
            'grad_direction': grad_direction.detach().clone()
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self):
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_queries': total,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size
        }

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"TEC Context Cache Statistics")
        print(f"{'='*60}")
        print(f"  Cache hits: {stats['hits']}")
        print(f"  Cache misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']*100:.2f}%")
        print(f"  Current cache size: {stats['cache_size']}/{stats['max_cache_size']}")
        print(f"{'='*60}\n")


# ======================== 测试代码 ========================
if __name__ == '__main__':
    print("Testing HourlyTECContextCache...")

    # 模拟ConvLSTM编码器（需要training属性）
    class DummyConvLSTM:
        def __init__(self):
            self.training = False  # 默认评估模式

        def __call__(self, x):
            # [N, Seq, 1, H, W] -> [N, 16, H, W]
            return torch.randn(x.shape[0], 16, x.shape[3], x.shape[4])

    class DummyGradHead:
        def __call__(self, x):
            # [N, 16, H, W] -> [N, 2, H, W]
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3])

    class DummyTECManager:
        def get_tec_map_sequence_by_hours(self, hour_indices):
            # 返回 [N_hours, Seq, 1, H, W]
            return torch.randn(len(hour_indices), 6, 1, 73, 73)

    # 创建缓存
    convlstm = DummyConvLSTM()
    grad_head = DummyGradHead()
    cache = HourlyTECContextCache(convlstm, grad_head, max_cache_size=10)

    tec_manager = DummyTECManager()

    # 测试1: 首次查询（全部未命中）
    print("\n测试1: 首次查询 [0, 1, 2]")
    hour_indices = [0, 1, 2]
    F_tec_dict, grad_dict = cache.get(hour_indices, tec_manager)
    cache.print_stats()

    # 测试2: 重复查询（全部命中）
    print("\n测试2: 重复查询 [0, 1, 2]")
    F_tec_dict, grad_dict = cache.get(hour_indices, tec_manager)
    cache.print_stats()

    # 测试3: 部分命中
    print("\n测试3: 部分命中 [1, 2, 3, 4]")
    hour_indices = [1, 2, 3, 4]
    F_tec_dict, grad_dict = cache.get(hour_indices, tec_manager)
    cache.print_stats()

    # 测试4: LRU淘汰
    print("\n测试4: LRU淘汰（max_cache_size=10）")
    hour_indices = list(range(15))  # 查询15个小时
    F_tec_dict, grad_dict = cache.get(hour_indices, tec_manager)
    cache.print_stats()

    # 验证最旧的小时被淘汰
    print(f"\n缓存中的小时: {list(cache.cache.keys())}")

    # 测试5: 训练模式下跳过缓存
    print("\n测试5: 训练模式（跳过缓存）")
    convlstm.training = True  # 切换到训练模式
    cache.clear()  # 清空缓存和统计
    hour_indices = [0, 1, 2]
    F_tec_dict, grad_dict = cache.get(hour_indices, tec_manager)
    cache.print_stats()
    print(f"训练模式下缓存大小应为0: {len(cache.cache)}")

    # 再次查询相同的小时，仍然不应使用缓存
    F_tec_dict, grad_dict = cache.get(hour_indices, tec_manager)
    cache.print_stats()
    print(f"重复查询后缓存大小仍为0: {len(cache.cache)}")

    print("\n✅ 所有测试通过!")

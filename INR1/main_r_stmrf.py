"""
R-STMRF 主程序入口

提供训练、评估和可视化的完整流程
"""

import os
import sys

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import torch

from inr_modules.r_stmrf.config_r_stmrf import get_config_r_stmrf, print_config_r_stmrf
from inr_modules.r_stmrf.train_r_stmrf import train_r_stmrf


def main():
    """主函数"""
    print("\n" + "="*70)
    print("R-STMRF: 物理引导的循环时空调制残差场")
    print("Recurrent Spatio-Temporal Modulated Residual Field")
    print("电离层电子密度三维重构")
    print("="*70 + "\n")

    # 获取配置
    config = get_config_r_stmrf()
    print_config_r_stmrf()

    # ==================== 1. 训练模型 ====================
    print("\n" + "="*70)
    print("阶段 1: 模型训练")
    print("="*70 + "\n")

    model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager = train_r_stmrf(config)

    device = torch.device(config['device'])

    # ==================== 2. 评估模型 ====================
    print("\n" + "="*70)
    print("阶段 2: 模型评估")
    print("="*70 + "\n")

    # 加载最佳模型
    best_model_path = os.path.join(config['save_dir'], 'best_r_stmrf_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✓ 已加载最佳模型: {best_model_path}")
    else:
        print("⚠️  警告: 未找到最佳模型，使用当前模型")

    # TODO: 实现详细评估（可复用原有的 evaluate_and_save_report）
    print("详细评估功能待实现...")

    # ==================== 3. 可视化结果 ====================
    print("\n" + "="*70)
    print("阶段 3: 可视化")
    print("="*70 + "\n")

    # 绘制损失曲线
    print("[3.1] 绘制损失曲线...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('R-STMRF Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_curve_path = os.path.join(config['save_dir'], 'loss_curve_r_stmrf.png')
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 损失曲线已保存: {loss_curve_path}")

    # TODO: 实现 Parity 图和高度剖面（可复用原有函数）
    print("\n[3.2] Parity 图和高度剖面功能待实现...")

    # ==================== 4. 完成 ====================
    print("\n" + "="*70)
    print("所有任务完成!")
    print(f"结果保存在: {os.path.abspath(config['save_dir'])}")
    print("="*70 + "\n")

    print("生成的文件:")
    print("  - best_r_stmrf_model.pth        (最佳模型权重)")
    print("  - loss_curve_r_stmrf.png        (损失曲线)")
    print("  - r_stmrf_epoch_*.pth           (定期保存的检查点)")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断训练")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()

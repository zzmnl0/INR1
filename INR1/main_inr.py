"""
主程序入口
提供训练、评估和可视化的完整流程
"""
import os
import sys

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
from inr_modules.config import get_config, print_config
from inr_modules.train import train_model
from inr_modules.utils import (
    evaluate_and_save_report,
    plot_loss_curve,
    evaluate_parity,
    plot_full_altitude_profile
)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("物理引导的隐式神经辐射场 (Physics-Guided INR)")
    print("电离层电子密度三维重构")
    print("="*70 + "\n")
    
    # 获取配置
    config = get_config()
    print_config()
    
    # ==================== 1. 训练模型 ====================
    print("\n" + "="*70)
    print("阶段 1: 模型训练")
    print("="*70 + "\n")
    
    model, train_losses, val_losses, train_loader, val_loader, sw_manager, tec_manager = train_model(config)
    
    device = torch.device(config['device'])
    
    # ==================== 2. 评估模型 ====================
    print("\n" + "="*70)
    print("阶段 2: 模型评估")
    print("="*70 + "\n")
    
    # 计算详细指标并保存报告
    evaluate_and_save_report(
        model, train_loader, val_loader,
        sw_manager, tec_manager, device,
        config['save_dir']
    )
    
    # ==================== 3. 可视化结果 ====================
    print("\n" + "="*70)
    print("阶段 3: 可视化")
    print("="*70 + "\n")
    
    # 绘制损失曲线
    print("[3.1] 绘制损失曲线...")
    plot_loss_curve(
        train_losses, val_losses,
        os.path.join(config['save_dir'], 'loss_curve_model_VTEC.png')
    )
    
    # 绘制Parity图
    print("\n[3.2] 绘制Parity图...")
    evaluate_parity(
        model, val_loader, sw_manager, tec_manager,
        device, config['save_dir'], config
    )
    
    # 绘制高度剖面
    print("\n[3.3] 绘制高度剖面...")
    plot_days = [5, 15, 25]
    target_hours = [0.0, 6.0, 12.0, 18.0]
    
    print(f"目标日期: {plot_days}")
    print("列: [IRI背景] | [INR预测] | [差异]\n")
    
    for day in plot_days:
        print(f"--- 处理 Day {day} ---")
        for hour in target_hours:
            plot_full_altitude_profile(
                model, sw_manager, tec_manager, device,
                target_day=day, target_hour=hour,
                save_dir=config['save_dir']
            )
    
    # ==================== 4. 完成 ====================
    print("\n" + "="*70)
    print("所有任务完成!")
    print(f"结果保存在: {os.path.abspath(config['save_dir'])}")
    print("="*70 + "\n")
    
    print("生成的文件:")
    print("  - best_model_vtec_mod.pth           (最佳模型权重)")
    print("  - evaluation_report.txt             (评估报告)")
    print("  - loss_curve_model_VTEC.png         (损失曲线)")
    print("  - parity_scatter_vtec.png           (散点图)")
    print("  - parity_density_vtec.png           (密度图)")
    print("  - profile_day*_hour*_vtec_mod.png   (高度剖面)")
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

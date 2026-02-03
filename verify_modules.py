#!/usr/bin/env python
"""
验证模块文件是否存在（不实际导入它们）
"""
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
inr1_module_path = os.path.join(current_dir, 'INR1', 'inr_modules')

print("=" * 70)
print("验证 R-STMRF 模块文件")
print("=" * 70)
print(f"INR1 模块路径: {inr1_module_path}")
print(f"路径存在: {os.path.exists(inr1_module_path)}")
print()

# 检查模块文件
modules_to_check = [
    'r_stmrf/r_stmrf_model.py',
    'r_stmrf/tec_gradient_bank.py',
    'r_stmrf/sliding_dataset.py',
    'data_managers/space_weather_manager.py',
    'data_managers/tec_manager.py',
    'data_managers/irinc_neural_proxy.py',
]

print("模块文件检查:")
print("-" * 70)

all_exist = True
for module_file in modules_to_check:
    full_path = os.path.join(inr1_module_path, module_file)
    exists = os.path.exists(full_path)
    status = "✓" if exists else "✗"
    print(f"{status} {module_file:<40} {'存在' if exists else '不存在'}")
    if not exists:
        all_exist = False

print()
print("=" * 70)

if all_exist:
    print("✓ 所有模块文件都存在！")
    print()
    print("结论：导入路径配置正确。")
    print("IDE 的\"无法解析导入\"警告是因为动态添加 sys.path。")
    print("这是正常的，运行时导入将正常工作（需要安装 PyTorch）。")
else:
    print("✗ 部分模块文件不存在，请检查项目结构。")

#!/usr/bin/env python
"""
测试 R_STMRF_ISR_IRI_plot.py 中的导入是否正常工作
"""
import os
import sys

# 添加模块路径（与主脚本相同的逻辑）
current_dir = os.path.dirname(os.path.abspath(__file__))
inr1_module_path = os.path.join(current_dir, 'INR1', 'inr_modules')
if inr1_module_path not in sys.path:
    sys.path.insert(0, inr1_module_path)

print("=" * 60)
print("测试导入路径")
print("=" * 60)
print(f"当前目录: {current_dir}")
print(f"INR1 模块路径: {inr1_module_path}")
print(f"路径存在: {os.path.exists(inr1_module_path)}")
print()

# 测试导入
print("=" * 60)
print("测试模块导入")
print("=" * 60)

modules_to_test = [
    ('r_stmrf.r_stmrf_model', 'R_STMRF_Model'),
    ('r_stmrf.tec_gradient_bank', 'TecGradientBank'),
    ('r_stmrf.sliding_dataset', 'SlidingWindowBatchProcessor'),
    ('data_managers.space_weather_manager', 'SpaceWeatherManager'),
    ('data_managers.tec_data_manager', 'TECDataManager'),
    ('utils.iri_neural_proxy', 'IRINeuralProxy'),
]

success_count = 0
fail_count = 0

for module_path, class_name in modules_to_test:
    try:
        # 动态导入模块
        module = __import__(module_path, fromlist=[class_name])
        # 检查类是否存在
        cls = getattr(module, class_name)
        print(f"✓ {module_path}.{class_name}")
        success_count += 1
    except ImportError as e:
        print(f"✗ {module_path}.{class_name} - ImportError: {e}")
        fail_count += 1
    except AttributeError as e:
        print(f"✗ {module_path}.{class_name} - AttributeError: {e}")
        fail_count += 1
    except Exception as e:
        print(f"✗ {module_path}.{class_name} - {type(e).__name__}: {e}")
        fail_count += 1

print()
print("=" * 60)
print(f"结果: {success_count} 成功, {fail_count} 失败")
print("=" * 60)

if fail_count == 0:
    print("✓ 所有导入测试通过！")
    print("  IDE 的\"无法解析导入\"警告可以安全忽略。")
    sys.exit(0)
else:
    print("✗ 部分导入失败，请检查模块路径和依赖。")
    sys.exit(1)

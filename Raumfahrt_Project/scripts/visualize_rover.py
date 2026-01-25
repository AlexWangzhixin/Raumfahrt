#!/usr/bin/env python3
"""
月球车3D可视化演示脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dynamics.rover_model import RoverModel
import importlib.util
import os

# 直接导入 visualization.py 文件
file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'visualization.py')
spec = importlib.util.spec_from_file_location("visualization", file_path)
visualization_module = importlib.util.module_from_spec(spec)
sys.modules["visualization"] = visualization_module
spec.loader.exec_module(visualization_module)
Visualization = visualization_module.Visualization

def main():
    """
    主函数
    """
    print("月球车3D可视化演示")
    print("=" * 50)
    
    # 创建月球车模型
    rover = RoverModel()
    print("✓ 月球车模型创建完成")
    
    # 获取月球车参数
    params = rover.get_rover_parameters()
    print("月球车参数:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    print()
    
    # 创建可视化对象
    viz = Visualization()
    print("✓ 可视化对象创建完成")
    
    # 选择可视化模式
    print("请选择可视化模式:")
    print("1. 仅月球车3D模型")
    print("2. 月球车在月壤环境中")
    choice = input("请输入选择 (1/2): ")
    print()
    
    if choice == '1':
        # 绘制月球车3D模型
        print("绘制月球车3D模型...")
        plotter = viz.plot_rover_3d(rover)
        print("✓ 月球车3D模型绘制完成")
    elif choice == '2':
        # 绘制完整的月球场景
        print("绘制月球场景...")
        print("生成月球表面地形...")
        plotter = viz.plot_lunar_scene(rover)
        print("✓ 月球场景绘制完成")
    else:
        print("无效选择，默认使用月球车3D模型")
        plotter = viz.plot_rover_3d(rover)
    
    # 显示可视化结果
    print("显示可视化结果...")
    print("提示: 可以使用鼠标交互操作:")
    print("  - 左键拖动: 旋转视角")
    print("  - 中键拖动: 平移视角")
    print("  - 滚轮: 缩放视角")
    print("  - 按 'q' 键退出")
    print()
    
    plotter.show()
    
    print("演示结束")

if __name__ == "__main__":
    main()

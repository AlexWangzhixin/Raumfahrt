#!/usr/bin/env python3
"""
月球车可视化测试用例
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dynamics.rover_model import RoverModel
import importlib.util
import sys

# 直接导入 visualization.py 文件
import os
file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'visualization.py')
spec = importlib.util.spec_from_file_location("visualization", file_path)
visualization_module = importlib.util.module_from_spec(spec)
sys.modules["visualization"] = visualization_module
spec.loader.exec_module(visualization_module)
Visualization = visualization_module.Visualization

class TestRoverVisualization(unittest.TestCase):
    """
    月球车可视化测试类
    """
    
    def test_rover_model_creation(self):
        """
        测试月球车模型创建
        """
        rover = RoverModel()
        self.assertIsNotNone(rover)
    
    def test_rover_3d_model_creation(self):
        """
        测试月球车3D模型创建
        """
        rover = RoverModel()
        model = rover.create_3d_model()
        self.assertIsNotNone(model)
    
    def test_visualization_3d_capability(self):
        """
        测试可视化模块的3D能力
        """
        viz = Visualization()
        # 检查是否有3D可视化方法
        self.assertTrue(hasattr(viz, 'plot_rover_3d'))
    
    def test_rover_visualization_integration(self):
        """
        测试月球车可视化集成
        """
        rover = RoverModel()
        viz = Visualization()
        # 检查是否能创建可视化
        try:
            fig = viz.plot_rover_3d(rover)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"可视化失败: {str(e)}")
    
    def test_lunar_terrain_creation(self):
        """
        测试月球表面地形创建
        """
        viz = Visualization()
        try:
            terrain = viz.create_lunar_terrain()
            self.assertIsNotNone(terrain)
        except Exception as e:
            self.fail(f"地形创建失败: {str(e)}")
    
    def test_lunar_scene_integration(self):
        """
        测试完整的月球场景集成
        """
        rover = RoverModel()
        viz = Visualization()
        try:
            plotter = viz.plot_lunar_scene(rover)
            self.assertIsNotNone(plotter)
        except Exception as e:
            self.fail(f"场景集成失败: {str(e)}")

if __name__ == '__main__':
    unittest.main()

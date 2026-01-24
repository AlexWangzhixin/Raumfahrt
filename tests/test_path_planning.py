# -*- coding: utf-8 -*-
"""
路径规划系统测试用例
"""

import unittest
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.planning import MultiScalePlanner, WaypointInterpolator, TerrainAwarePlanner, TerrainModel


class TestMultiScalePlanner(unittest.TestCase):
    """测试多尺度路径规划器"""

    def setUp(self):
        """设置测试环境"""
        self.planner = MultiScalePlanner()
        # 添加测试地图
        self.obstacles = [
            (20.0, 20.0, 2.0),
            (50.0, 50.0, 3.0),
            (80.0, 80.0, 2.0)
        ]
        self.planner.add_map(1.0, self.obstacles, 100.0, 100.0)
        self.planner.add_map(5.0, self.obstacles, 100.0, 100.0)

    def test_plan_global_path(self):
        """测试全局路径规划"""
        start = (10.0, 10.0)
        goal = (90.0, 90.0)
        
        path = self.planner.plan_global_path(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        print(f"全局路径规划测试通过，路径点数量: {len(path)}")

    def test_get_appropriate_resolution(self):
        """测试分辨率选择"""
        # 测试短距离
        resolution_short = self.planner.get_appropriate_resolution(5.0)
        self.assertLess(resolution_short, 1.0)
        
        # 测试中等距离
        resolution_medium = self.planner.get_appropriate_resolution(100.0)
        self.assertEqual(resolution_medium, 1.0)
        
        # 测试长距离
        resolution_long = self.planner.get_appropriate_resolution(1500.0)
        self.assertEqual(resolution_long, 10.0)
        print("分辨率选择测试通过")

    def test_sparsify_path(self):
        """测试路径点稀疏化"""
        # 创建测试路径
        test_path = [(i, i) for i in range(0, 100, 5)]
        sparse_path = self.planner._sparsify_path(test_path, 25.0)
        self.assertLess(len(sparse_path), len(test_path))
        print(f"路径稀疏化测试通过，从 {len(test_path)} 点稀疏到 {len(sparse_path)} 点")

    def test_get_local_waypoints(self):
        """测试获取局部路径点"""
        # 创建测试路径
        test_path = [(i, i) for i in range(0, 100, 10)]
        current_pos = (35.0, 35.0)
        local_waypoints = self.planner.get_local_waypoints(test_path, current_pos, look_ahead=2)
        self.assertEqual(len(local_waypoints), 3)  # 包括当前点和两个前瞻点
        print(f"局部路径点获取测试通过，获取到 {len(local_waypoints)} 个局部路径点")


class TestWaypointInterpolator(unittest.TestCase):
    """测试路径点插值器"""

    def setUp(self):
        """设置测试环境"""
        self.interpolator = WaypointInterpolator()

    def test_interpolate_path(self):
        """测试路径插值"""
        # 创建稀疏路径
        sparse_path = [(0, 0), (10, 10), (20, 20)]
        interpolated_path = self.interpolator.interpolate_path(sparse_path, 2.0)
        self.assertGreater(len(interpolated_path), len(sparse_path))
        print(f"路径插值测试通过，从 {len(sparse_path)} 点插值到 {len(interpolated_path)} 点")

    def test_smooth_path(self):
        """测试路径平滑"""
        # 创建测试路径
        test_path = [(0, 0), (5, 10), (10, 0), (15, 10), (20, 0)]
        smooth_path = self.interpolator.smooth_path(test_path)
        self.assertEqual(len(smooth_path), len(test_path))
        print("路径平滑测试通过")


class TestTerrainModel(unittest.TestCase):
    """测试地形模型"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试地形数据
        self.terrain_data = np.zeros((50, 50))
        for i in range(50):
            for j in range(50):
                self.terrain_data[i, j] = 0.1 * i + 0.05 * j
        self.terrain_model = TerrainModel(self.terrain_data, 1.0, (0.0, 0.0))

    def test_get_height(self):
        """测试获取地形高度"""
        height = self.terrain_model.get_height(10.0, 10.0)
        self.assertAlmostEqual(height, 1.5)
        print(f"地形高度获取测试通过，高度: {height:.2f}")

    def test_get_slope(self):
        """测试获取地形坡度"""
        slope = self.terrain_model.get_slope(20.0, 20.0)
        self.assertAlmostEqual(slope, 0.1118, places=4)  # arctan(0.1118) ≈ 6.4度
        print(f"地形坡度获取测试通过，坡度: {np.degrees(slope):.2f}度")

    def test_is_traversable(self):
        """测试地形可通行性"""
        # 测试平坦地形
        self.assertTrue(self.terrain_model.is_traversable(10.0, 10.0))
        # 测试陡坡（创建一个陡坡）
        steep_terrain = np.zeros((50, 50))
        for i in range(50):
            steep_terrain[i, :] = 2.0 * i  # 非常陡的坡
        steep_model = TerrainModel(steep_terrain, 1.0, (0.0, 0.0))
        self.assertFalse(steep_model.is_traversable(20.0, 20.0))
        print("地形可通行性测试通过")


class TestTerrainAwarePlanner(unittest.TestCase):
    """测试地形感知路径规划器"""

    def setUp(self):
        """设置测试环境"""
        # 创建多尺度规划器
        self.multi_scale_planner = MultiScalePlanner()
        obstacles = [
            (20.0, 20.0, 2.0),
            (80.0, 80.0, 3.0)
        ]
        self.multi_scale_planner.add_map(1.0, obstacles, 100.0, 100.0)
        
        # 创建地形模型
        terrain_data = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                terrain_data[i, j] = 0.1 * np.sin(i/10) * np.cos(j/10)
                # 添加一个陡坡区域
                if 40 <= i <= 60 and 40 <= j <= 60:
                    terrain_data[i, j] = 0.5 * ((i-50)**2 + (j-50)**2)**0.5 / 10
        self.terrain_model = TerrainModel(terrain_data, 1.0, (0.0, 0.0))
        
        # 创建地形感知规划器
        self.terrain_planner = TerrainAwarePlanner(self.multi_scale_planner, self.terrain_model)

    def test_plan_terrain_aware_path(self):
        """测试地形感知路径规划"""
        start = (10.0, 10.0)
        goal = (90.0, 90.0)
        
        path = self.terrain_planner.plan_terrain_aware_path(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        print(f"地形感知路径规划测试通过，路径点数量: {len(path)}")

    def test_evaluate_path(self):
        """测试路径评估"""
        # 创建测试路径
        test_path = [(i, i) for i in range(0, 100, 10)]
        evaluation = self.terrain_planner.evaluate_path(test_path)
        
        self.assertIn('length', evaluation)
        self.assertIn('average_slope', evaluation)
        self.assertIn('max_slope', evaluation)
        self.assertIn('height_changes', evaluation)
        self.assertIn('traversability', evaluation)
        
        print("路径评估测试通过")
        print(f"  路径长度: {evaluation['length']:.2f} 米")
        print(f"  平均坡度: {np.degrees(evaluation['average_slope']):.2f} 度")
        print(f"  最大坡度: {np.degrees(evaluation['max_slope']):.2f} 度")
        print(f"  可通行性得分: {evaluation['traversability']:.2f}")

    def test_find_nearest_safe_point(self):
        """测试寻找最近的安全点"""
        # 测试安全点
        safe_point = (10.0, 10.0)
        found_point = self.terrain_planner._find_nearest_safe_point(safe_point)
        self.assertEqual(found_point, safe_point)
        
        # 测试危险点（陡坡区域）
        dangerous_point = (50.0, 50.0)
        found_safe_point = self.terrain_planner._find_nearest_safe_point(dangerous_point)
        self.assertIsNotNone(found_safe_point)
        distance = np.sqrt((found_safe_point[0] - dangerous_point[0])**2 + 
                         (found_safe_point[1] - dangerous_point[1])**2)
        self.assertLess(distance, 20.0)  # 应该在20米内找到安全点
        print("寻找安全点测试通过")


if __name__ == '__main__':
    print("开始路径规划系统测试...")
    unittest.main()

#!/usr/bin/env python3
"""
测试轨迹生成器
验证缓动轨迹生成功能
"""

import unittest
import numpy as np


class TestTrajectoryGenerator(unittest.TestCase):
    """测试轨迹生成器类"""
    
    def setUp(self):
        """设置测试环境"""
        from src.core.planning.trajectory_generator import TrajectoryGenerator
        self.generator = TrajectoryGenerator()
        self.start_pos = (100, 100)
        self.end_pos = (100, 500)
        self.duration = 10.0
        self.fps = 50
    
    def test_ease_in_out_cubic(self):
        """测试三次缓动函数"""
        # 测试边界值
        self.assertAlmostEqual(self.generator.ease_in_out_cubic(0.0), 0.0)
        self.assertAlmostEqual(self.generator.ease_in_out_cubic(1.0), 1.0)
        
        # 测试中间值
        self.assertAlmostEqual(self.generator.ease_in_out_cubic(0.5), 0.5)
        
        # 测试缓动特性
        t_values = np.linspace(0, 1, 100)
        eased_values = [self.generator.ease_in_out_cubic(t) for t in t_values]
        
        # 确保缓动函数单调递增
        for i in range(1, len(eased_values)):
            self.assertGreaterEqual(eased_values[i], eased_values[i-1])
    
    def test_generate_smooth_straight_line(self):
        """测试生成平滑直线轨迹"""
        # 生成轨迹
        trajectory = self.generator.generate_smooth_straight_line(
            self.start_pos, 
            self.end_pos, 
            self.duration, 
            self.fps
        )
        
        # 验证轨迹形状
        self.assertIsInstance(trajectory, np.ndarray)
        self.assertEqual(len(trajectory), int(self.duration * self.fps))
        self.assertEqual(trajectory.shape[1], 2)
        
        # 验证起点和终点
        np.testing.assert_array_almost_equal(trajectory[0], self.start_pos)
        np.testing.assert_array_almost_equal(trajectory[-1], self.end_pos)
        
        # 验证轨迹是垂直直线（X坐标不变）
        all_x = trajectory[:, 0]
        x_variation = np.max(all_x) - np.min(all_x)
        self.assertAlmostEqual(x_variation, 0.0)
        
        # 验证Y坐标从100到500递增
        all_y = trajectory[:, 1]
        self.assertEqual(all_y[0], 100.0)
        self.assertEqual(all_y[-1], 500.0)
        for i in range(1, len(all_y)):
            self.assertGreaterEqual(all_y[i], all_y[i-1])
        
        # 验证中间点（考虑缓动函数特性，允许一定误差）
        middle_idx = len(trajectory) // 2
        middle_point = trajectory[middle_idx]
        expected_middle = (100, 300)  # 中间点理论值
        # 允许±2的误差，因为缓动函数在中间点附近的特性
        np.testing.assert_array_almost_equal(middle_point, expected_middle, decimal=0)
    
    def test_trajectory_smoothness(self):
        """测试轨迹平滑度"""
        # 生成轨迹
        trajectory = self.generator.generate_smooth_straight_line(
            self.start_pos, 
            self.end_pos, 
            self.duration, 
            self.fps
        )
        
        # 计算速度
        velocities = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i, 0] - trajectory[i-1, 0]
            dy = trajectory[i, 1] - trajectory[i-1, 1]
            distance = np.sqrt(dx**2 + dy**2)
            time_interval = 1.0 / self.fps
            velocity = distance / time_interval
            velocities.append(velocity)
        
        # 验证速度曲线平滑（没有突变）
        for i in range(1, len(velocities)):
            velocity_change = abs(velocities[i] - velocities[i-1])
            # 速度变化应该小于阈值
            self.assertLess(velocity_change, 0.18)  # 0.18 m/s 阈值，允许微小的计算误差
        
        # 验证速度曲线符合缓动特性（先加速后减速）
        # 找到速度最大值的位置
        max_velocity_idx = np.argmax(velocities)
        # 最大值应该在中间附近
        self.assertGreater(max_velocity_idx, len(velocities) * 0.3)
        self.assertLess(max_velocity_idx, len(velocities) * 0.7)


if __name__ == '__main__':
    unittest.main()

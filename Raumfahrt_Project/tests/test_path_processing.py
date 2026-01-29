#!/usr/bin/env python3
"""
路径处理模块测试用例
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入路径处理模块
try:
    from src.core.path_processing import smooth_path_bspline, load_real_path_data
except ImportError:
    # 如果模块不存在，测试会失败，这是预期的
    smooth_path_bspline = None
    load_real_path_data = None


class TestPathProcessing(unittest.TestCase):
    """路径处理模块测试类"""
    
    def test_smooth_path_bspline_basic(self):
        """测试基本的B-spline平滑功能"""
        # 准备测试数据
        x = np.array([0.0, 5.0, 12.0, 18.0, 28.0, 32.0])
        y = np.array([0.0, 2.0, 1.5, 4.0, 3.0, 0.0])
        
        # 测试函数调用
        result = smooth_path_bspline(x, y, num_points=10)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 2))
        self.assertTrue(np.allclose(result[0], [0.0, 0.0]))  # 起点不变
        self.assertTrue(np.allclose(result[-1], [32.0, 0.0]))  # 终点不变
    
    def test_smooth_path_bspline_insufficient_points(self):
        """测试点数不足时的线性插值处理"""
        # 准备测试数据（只有2个点）
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 5.0])
        
        # 测试函数调用
        result = smooth_path_bspline(x, y, num_points=5)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(np.allclose(result[0], [0.0, 0.0]))
        self.assertTrue(np.allclose(result[-1], [10.0, 5.0]))
    
    def test_smooth_path_bspline_duplicate_points(self):
        """测试处理重复点的情况"""
        # 准备测试数据（包含重复点）
        x = np.array([0.0, 5.0, 5.0, 10.0])
        y = np.array([0.0, 2.0, 2.0, 0.0])
        
        # 测试函数调用
        result = smooth_path_bspline(x, y, num_points=8)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (8, 2))
    
    def test_load_real_path_data_nonexistent_file(self):
        """测试加载不存在的路径文件时的处理"""
        # 使用不存在的文件路径
        nonexistent_file = "nonexistent_path.csv"
        
        # 测试函数调用
        result = load_real_path_data(nonexistent_file)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.shape[0] > 0)  # 应该返回模拟数据


if __name__ == '__main__':
    unittest.main()

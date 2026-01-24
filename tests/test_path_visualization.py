import unittest
import numpy as np
import os
import sys
from unittest.mock import patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.visualization.path_visualization import PathVisualizer

class TestPathVisualization(unittest.TestCase):
    """
    路径可视化功能测试
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        self.visualizer = PathVisualizer()
        
        # 创建测试路径
        self.test_path = [(0, 0), (10, 5), (20, 8), (30, 3), (40, 10), (50, 0)]
        
        # 创建测试障碍物
        self.test_obstacles = [
            {'x': 15, 'y': 12, 'radius': 2.0},
            {'x': 35, 'y': 5, 'radius': 1.5},
            {'x': 25, 'y': 0, 'radius': 1.0}
        ]
        
        # 创建测试关键节点
        self.test_key_nodes = [0, 2, 4, 5]
        
        # 创建测试处理步骤
        self.test_process_steps = [
            '起点',
            '环境建模',
            '动力学仿真',
            '感知处理',
            '路径规划',
            '终点'
        ]
        
        # 创建测试地形数据
        self.test_terrain_data = {
            'height_map': np.random.rand(100, 100) * 10 - 5,  # 随机高度图
            'extent': [-10, 60, -10, 20]
        }
        
        # 测试输出目录
        self.output_dir = 'data/visualizations/path'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_initialization(self):
        """
        测试路径可视化工具的初始化
        """
        self.assertIsInstance(self.visualizer, PathVisualizer)
        self.assertIsInstance(self.visualizer.colors, dict)
        self.assertIsInstance(self.visualizer.node_sizes, dict)
        self.assertIsInstance(self.visualizer.line_styles, dict)
        self.assertIsInstance(self.visualizer.line_widths, dict)
    
    def test_visualize_topological_map(self):
        """
        测试拓扑图可视化
        """
        from matplotlib.figure import Figure
        
        # 测试基本功能
        fig = self.visualizer.visualize_topological_map(self.test_path)
        self.assertIsInstance(fig, Figure)
        
        # 测试带障碍物的功能
        fig_with_obstacles = self.visualizer.visualize_topological_map(
            self.test_path, 
            obstacles=self.test_obstacles
        )
        self.assertIsInstance(fig_with_obstacles, Figure)
        
        # 测试带关键节点的功能
        fig_with_key_nodes = self.visualizer.visualize_topological_map(
            self.test_path, 
            obstacles=self.test_obstacles,
            key_nodes=self.test_key_nodes
        )
        self.assertIsInstance(fig_with_key_nodes, Figure)
        
        # 测试空路径
        fig_empty = self.visualizer.visualize_topological_map([])
        self.assertIsInstance(fig_empty, Figure)
        
        # 关闭图形
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_visualize_flowchart(self):
        """
        测试流程图可视化
        """
        from matplotlib.figure import Figure
        
        # 测试基本功能
        fig = self.visualizer.visualize_flowchart(self.test_path)
        self.assertIsInstance(fig, Figure)
        
        # 测试带处理步骤的功能
        fig_with_steps = self.visualizer.visualize_flowchart(
            self.test_path, 
            process_steps=self.test_process_steps
        )
        self.assertIsInstance(fig_with_steps, Figure)
        
        # 测试空路径
        fig_empty = self.visualizer.visualize_flowchart([])
        self.assertIsInstance(fig_empty, Figure)
        
        # 关闭图形
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_visualize_geographic_map(self):
        """
        测试地理分布图可视化
        """
        from matplotlib.figure import Figure
        
        # 测试基本功能
        fig = self.visualizer.visualize_geographic_map(self.test_path)
        self.assertIsInstance(fig, Figure)
        
        # 测试带障碍物的功能
        fig_with_obstacles = self.visualizer.visualize_geographic_map(
            self.test_path, 
            obstacles=self.test_obstacles
        )
        self.assertIsInstance(fig_with_obstacles, Figure)
        
        # 测试带地形数据的功能
        fig_with_terrain = self.visualizer.visualize_geographic_map(
            self.test_path, 
            obstacles=self.test_obstacles,
            terrain_data=self.test_terrain_data
        )
        self.assertIsInstance(fig_with_terrain, Figure)
        
        # 测试空路径
        fig_empty = self.visualizer.visualize_geographic_map([])
        self.assertIsInstance(fig_empty, Figure)
        
        # 关闭图形
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_visualize_all_modes(self):
        """
        测试所有模式的可视化
        """
        from matplotlib.figure import Figure
        
        # 测试所有模式
        fig = self.visualizer.visualize_all_modes(
            self.test_path,
            obstacles=self.test_obstacles,
            key_nodes=self.test_key_nodes,
            process_steps=self.test_process_steps,
            terrain_data=self.test_terrain_data
        )
        self.assertIsInstance(fig, Figure)
        
        # 关闭图形
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_save_figure(self):
        """
        测试图像保存功能
        """
        from matplotlib.figure import Figure
        
        # 创建一个简单的图形
        fig = self.visualizer.visualize_topological_map(self.test_path)
        
        # 测试保存功能
        test_filename = 'test_save_figure'
        self.visualizer.save_figure(fig, test_filename, file_formats=['png', 'svg'])
        
        # 验证文件是否存在
        png_path = os.path.join(self.output_dir, f'{test_filename}.png')
        svg_path = os.path.join(self.output_dir, f'{test_filename}.svg')
        
        self.assertTrue(os.path.exists(png_path))
        self.assertTrue(os.path.exists(svg_path))
        
        # 验证文件大小
        self.assertGreater(os.path.getsize(png_path), 0)
        self.assertGreater(os.path.getsize(svg_path), 0)
        
        # 清理测试文件
        if os.path.exists(png_path):
            os.remove(png_path)
        if os.path.exists(svg_path):
            os.remove(svg_path)
    
    def test_visualize_path(self):
        """
        测试路径可视化主函数
        """
        # 测试保存模式
        test_filename = 'test_visualize_path'
        self.visualizer.visualize_path(
            self.test_path,
            obstacles=self.test_obstacles,
            key_nodes=self.test_key_nodes,
            process_steps=self.test_process_steps,
            terrain_data=self.test_terrain_data,
            save=True,
            filename=test_filename
        )
        
        # 验证文件是否存在
        png_path = os.path.join(self.output_dir, f'{test_filename}.png')
        svg_path = os.path.join(self.output_dir, f'{test_filename}.svg')
        
        self.assertTrue(os.path.exists(png_path))
        self.assertTrue(os.path.exists(svg_path))
        
        # 清理测试文件
        if os.path.exists(png_path):
            os.remove(png_path)
        if os.path.exists(svg_path):
            os.remove(svg_path)
        
        # 测试非保存模式（仅创建图形）
        with patch('matplotlib.pyplot.show') as mock_show:
            self.visualizer.visualize_path(
                self.test_path,
                save=False
            )
            # 不调用show，因为我们只是测试图形创建
    
    def test_visualize_interactive(self):
        """
        测试交互式可视化
        """
        # 测试交互式可视化的初始化
        with patch('matplotlib.pyplot.show') as mock_show:
            self.visualizer.visualize_interactive(
                self.test_path,
                obstacles=self.test_obstacles,
                key_nodes=self.test_key_nodes,
                terrain_data=self.test_terrain_data
            )
            # 不调用show，因为我们只是测试初始化
    
    def test_edge_cases(self):
        """
        测试边界情况
        """
        # 测试空路径
        with patch('matplotlib.pyplot.show') as mock_show:
            self.visualizer.visualize_path([])
        
        # 测试只有一个点的路径
        single_point_path = [(0, 0)]
        fig = self.visualizer.visualize_topological_map(single_point_path)
        self.assertIsNotNone(fig)
        
        # 测试只有两个点的路径
        two_points_path = [(0, 0), (10, 10)]
        fig = self.visualizer.visualize_topological_map(two_points_path)
        self.assertIsNotNone(fig)
        
        # 关闭图形
        import matplotlib.pyplot as plt
        plt.close('all')

if __name__ == '__main__':
    unittest.main()

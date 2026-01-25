#!/usr/bin/env python3
"""
统一绘图工具类
提供各种可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
from src.config.config import VISUALIZATION_PARAMS
import pyvista as pv
from scipy.ndimage import gaussian_filter

class Visualization:
    """
    可视化工具类
    提供轨迹、地图、动力学数据的可视化功能
    """
    
    def __init__(self, figsize=None, dpi=None):
        """
        初始化可视化工具
        
        Args:
            figsize: 图表大小，默认使用配置中的值
            dpi: 图表DPI，默认使用配置中的值
        """
        self.figsize = figsize or VISUALIZATION_PARAMS['FIGURE_SIZE']
        self.dpi = dpi or VISUALIZATION_PARAMS['DPI']
        self.color_map = VISUALIZATION_PARAMS['COLOR_MAP']
        
    def create_figure(self, subplots=(1, 1)):
        """
        创建图表
        
        Args:
            subplots: 子图布局 (rows, cols)
        
        Returns:
            fig, axes: 图表和轴对象
        """
        fig, axes = plt.subplots(
            subplots[0], subplots[1], 
            figsize=self.figsize, 
            dpi=self.dpi
        )
        return fig, axes
    
    def plot_trajectory(self, trajectory, ax=None, color=None, label='Trajectory'):
        """
        绘制轨迹
        
        Args:
            trajectory: 轨迹点列表 [(x1, y1), (x2, y2), ...]
            ax: 轴对象，如果为None则创建新图表
            color: 轨迹颜色，默认使用配置中的值
            label: 轨迹标签
        
        Returns:
            fig, ax: 图表和轴对象
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()
        
        color = color or VISUALIZATION_PARAMS['TRAJECTORY_COLOR']
        trajectory = np.array(trajectory)
        
        if len(trajectory) > 0:
            ax.plot(
                trajectory[:, 0], 
                trajectory[:, 1], 
                color=color, 
                linewidth=2, 
                label=label
            )
            # 绘制起点和终点
            ax.plot(
                trajectory[0, 0], 
                trajectory[0, 1], 
                'o', 
                color=VISUALIZATION_PARAMS['START_COLOR'], 
                markersize=8, 
                label='Start'
            )
            ax.plot(
                trajectory[-1, 0], 
                trajectory[-1, 1], 
                'o', 
                color=VISUALIZATION_PARAMS['GOAL_COLOR'], 
                markersize=8, 
                label='Goal'
            )
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax
    
    def plot_map(self, grid_map, ax=None, title='Map'):
        """
        绘制地图
        
        Args:
            grid_map: 二维数组，表示地图高度或障碍物信息
            ax: 轴对象，如果为None则创建新图表
            title: 图表标题
        
        Returns:
            fig, ax: 图表和轴对象
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()
        
        im = ax.imshow(
            grid_map, 
            cmap=self.color_map, 
            origin='lower'
        )
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Elevation (m)')
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title)
        
        return fig, ax
    
    def plot_dynamics_data(self, time_steps, data, labels, ax=None, title='Dynamics Data'):
        """
        绘制动力学数据
        
        Args:
            time_steps: 时间步列表
            data: 数据列表，每个元素是一个时间序列
            labels: 数据标签列表
            ax: 轴对象，如果为None则创建新图表
            title: 图表标题
        
        Returns:
            fig, ax: 图表和轴对象
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, (d, label) in enumerate(zip(data, labels)):
            color = colors[i % len(colors)]
            ax.plot(time_steps, d, color=color, linewidth=2, label=label)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax
    
    def plot_comparison(self, data1, data2, labels, ax=None, title='Comparison'):
        """
        绘制对比图表
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            labels: 标签列表 [label1, label2]
            ax: 轴对象，如果为None则创建新图表
            title: 图表标题
        
        Returns:
            fig, ax: 图表和轴对象
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()
        
        x = np.arange(len(data1))
        width = 0.35
        
        ax.bar(x - width/2, data1, width, label=labels[0])
        ax.bar(x + width/2, data2, width, label=labels[1])
        
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        return fig, ax
    
    def save_figure(self, fig, file_path, dpi=None):
        """
        保存图表
        
        Args:
            fig: 图表对象
            file_path: 保存路径
            dpi: 保存DPI，默认使用配置中的值
        """
        dpi = dpi or self.dpi
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def show(self):
        """
        显示图表
        """
        plt.show()
    
    def plot_rover_3d(self, rover_model, title='Lunar Rover 3D Model'):
        """
        绘制月球车3D模型
        
        Args:
            rover_model: RoverModel 对象
            title: 图表标题
        
        Returns:
            pyvista.Plotter: 可视化对象
        """
        # 创建3D模型
        model = rover_model.create_3d_model()
        
        # 创建绘图器
        plotter = pv.Plotter(window_size=[800, 600])
        
        # 添加月球车模型
        plotter.add_mesh(
            model,
            color='gray',
            smooth_shading=True,
            opacity=1.0,
            show_edges=False
        )
        
        # 添加坐标轴
        plotter.add_axes()
        
        # 设置标题
        plotter.add_title(title)
        
        # 设置相机位置
        plotter.camera_position = 'iso'
        
        return plotter
    
    def create_lunar_terrain(self, size=20, resolution=100, max_elevation=1.0):
        """
        创建月球表面地形
        
        Args:
            size: 地形大小 (m)
            resolution: 地形分辨率
            max_elevation: 最大地形高度 (m)
        
        Returns:
            pyvista.PolyData: 月球表面地形模型
        """
        # 创建网格
        x = np.linspace(-size/2, size/2, resolution)
        y = np.linspace(-size/2, size/2, resolution)
        x, y = np.meshgrid(x, y)
        
        # 生成月球表面地形（使用第三章的分形噪声算法）
        z = self._generate_fractal_terrain(x, y, max_elevation)
        
        # 创建地形网格
        grid = pv.StructuredGrid(x, y, z)
        terrain = grid.extract_surface()
        
        return terrain
    
    def _generate_fractal_terrain(self, x, y, max_elevation):
        """
        生成第三章中使用的分形噪声地形
        
        Args:
            x: x坐标网格
            y: y坐标网格
            max_elevation: 最大地形高度
        
        Returns:
            np.ndarray: 地形高度网格
        """
        height, width = x.shape
        
        # 已在文件顶部导入必要的库
        
        # 生成分形噪声地形
        terrain = self._generate_fractal_noise(height, width, octaves=4, persistence=0.6)
        
        # 添加地形特征
        terrain = self._add_terrain_features(terrain)
        
        # 归一化并缩放到指定高度范围
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
        terrain = terrain * max_elevation - max_elevation * 0.2
        
        return terrain
    
    def _generate_fractal_noise(self, height, width, octaves=3, persistence=0.5):
        """
        生成分形噪声，避免周期性
        
        Args:
            height: 高度
            width: 宽度
            octaves: 八度数量
            persistence: 持久性
        
        Returns:
            分形噪声数据
        """
        # 初始化噪声
        noise = np.zeros((height, width))
        
        # 分形噪声生成
        for octave in range(octaves):
            scale = 2 ** octave
            octave_noise = np.random.rand(height // scale + 1, width // scale + 1)
            
            # 上采样
            from skimage.transform import resize
            octave_noise_up = resize(octave_noise, (height, width), order=1)
            
            # 添加到噪声
            noise += octave_noise_up * (persistence ** octave)
        
        # 平滑处理
        noise = gaussian_filter(noise, sigma=1)
        
        return noise
    
    def _add_terrain_features(self, terrain):
        """
        添加地形特征
        
        Args:
            terrain: 基础地形数据
        
        Returns:
            带特征的地形数据
        """
        height, width = terrain.shape
        
        # 添加一些山丘
        for _ in range(3):
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(10, 30)
            
            for i in range(max(0, center_y-radius), min(height, center_y+radius)):
                for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                    distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                    if distance < radius:
                        terrain[i, j] += 0.1 * (1 - distance/radius)
        
        # 添加一些洼地
        for _ in range(2):
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(8, 20)
            
            for i in range(max(0, center_y-radius), min(height, center_y+radius)):
                for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                    distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                    if distance < radius:
                        terrain[i, j] -= 0.06 * (1 - distance/radius)
        
        # 添加一些小尺度特征
        for _ in range(10):
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(2, 6)
            
            for i in range(max(0, center_y-radius), min(height, center_y+radius)):
                for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                    distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                    if distance < radius:
                        terrain[i, j] += 0.02 * (1 - distance/radius)
        
        return terrain
    
    def plot_lunar_scene(self, rover_model, terrain_size=20, terrain_resolution=100, max_elevation=1.0, title='Lunar Rover on Surface'):
        """
        绘制完整的月球场景（月球车+月壤环境）
        
        Args:
            rover_model: RoverModel 对象
            terrain_size: 地形大小 (m)
            terrain_resolution: 地形分辨率
            max_elevation: 最大地形高度 (m)
            title: 图表标题
        
        Returns:
            pyvista.Plotter: 可视化对象
        """
        # 创建3D模型
        model = rover_model.create_3d_model()
        
        # 创建月球表面地形
        terrain = self.create_lunar_terrain(
            size=terrain_size,
            resolution=terrain_resolution,
            max_elevation=max_elevation
        )
        
        # 创建绘图器
        plotter = pv.Plotter(window_size=[1000, 800])
        
        # 添加月球表面地形
        plotter.add_mesh(
            terrain,
            color='tan',
            smooth_shading=True,
            opacity=1.0,
            show_edges=False,
            ambient=0.2,
            diffuse=0.8
        )
        
        # 添加月球车模型
        plotter.add_mesh(
            model,
            color='gray',
            smooth_shading=True,
            opacity=1.0,
            show_edges=False,
            ambient=0.2,
            diffuse=0.8
        )
        
        # 添加坐标轴
        plotter.add_axes()
        
        # 设置标题
        plotter.add_title(title)
        
        # 设置相机位置
        plotter.camera_position = 'iso'
        
        # 设置光照
        plotter.add_light(pv.Light(position=(50, 50, 50), intensity=1.0))
        
        return plotter

# 导出类
__all__ = ['Visualization']

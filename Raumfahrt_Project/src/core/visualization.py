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

# 导出类
__all__ = ['Visualization']

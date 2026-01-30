#!/usr/bin/env python3
"""
统一绘图工具类
提供标准化的绘图接口
"""

import os
from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """
    可视化工具类
    """
    
    def __init__(self):
        """
        初始化可视化工具
        """
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 设置默认风格
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def create_figure(self, figsize: Tuple[int, int] = (10, 8)) -> Tuple[Any, Any]:
        """
        创建图形对象
        
        Args:
            figsize: 图形大小 (width, height)
        
        Returns:
            (fig, ax): 图形对象和坐标轴对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def save_figure(self, fig: Any, path: str, dpi: int = 300) -> None:
        """
        保存图形
        
        Args:
            fig: 图形对象
            path: 保存路径
            dpi: 分辨率
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"图形已保存到: {path}")
    
    def plot_trajectory(self, 
                       trajectory: Any, 
                       map_data: Optional[Any] = None, 
                       title: str = "Trajectory", 
                       save_path: Optional[str] = None) -> None:
        """
        绘制轨迹
        
        Args:
            trajectory: 轨迹点列表
            map_data: 地图数据 (可选)
            title: 标题
            save_path: 保存路径 (可选)
        """
        fig, ax = self.create_figure()
        
        if map_data is not None:
            ax.imshow(map_data, cmap='gray', origin='lower', alpha=0.5)
        
        traj_array = np.array(trajectory)
        ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            self.save_figure(fig, save_path)
        else:
            plt.show()

# 导出类
__all__ = ['Visualization']

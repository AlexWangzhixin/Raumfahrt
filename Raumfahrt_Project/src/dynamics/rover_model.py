#!/usr/bin/env python3
"""
月球车模型类
提供月球车的物理模型和3D可视化功能
"""

import pyvista as pv
import numpy as np

class RoverModel:
    """
    月球车模型类
    """
    
    def __init__(self):
        """
        初始化月球车模型
        """
        # 月球车参数
        self.length = 2.0  # 长度 (m)
        self.width = 1.5   # 宽度 (m)
        self.height = 1.0  # 高度 (m)
        self.wheel_radius = 0.3  # 车轮半径 (m)
        self.wheel_width = 0.2   # 车轮宽度 (m)
        self.wheel_base = 1.2    # 轴距 (m)
        self.wheel_track = 1.3   # 轮距 (m)
    
    def create_3d_model(self):
        """
        创建月球车3D模型
        
        Returns:
            pyvista.PolyData: 月球车3D模型
        """
        # 创建月球车主体
        body = self._create_body()
        
        # 创建车轮
        wheels = self._create_wheels()
        
        # 合并所有部件
        rover_model = body
        for wheel in wheels:
            rover_model += wheel
        
        return rover_model
    
    def _create_body(self):
        """
        创建月球车主体
        
        Returns:
            pyvista.PolyData: 月球车主体
        """
        # 创建主体长方体
        body = pv.Cube(
            center=(0, 0, self.height / 2),
            x_length=self.length,
            y_length=self.width,
            z_length=self.height
        )
        
        # 创建车顶太阳能板
        solar_panel = pv.Cube(
            center=(0, 0, self.height + 0.05),
            x_length=self.length * 0.9,
            y_length=self.width * 0.9,
            z_length=0.1
        )
        
        # 合并主体和太阳能板
        body += solar_panel
        
        return body
    
    def _create_wheels(self):
        """
        创建月球车车轮
        
        Returns:
            list: 车轮模型列表
        """
        wheels = []
        
        # 车轮位置
        wheel_positions = [
            (self.wheel_base / 2, self.wheel_track / 2, self.wheel_radius),  # 右前轮
            (self.wheel_base / 2, -self.wheel_track / 2, self.wheel_radius),  # 左前轮
            (-self.wheel_base / 2, self.wheel_track / 2, self.wheel_radius),  # 右后轮
            (-self.wheel_base / 2, -self.wheel_track / 2, self.wheel_radius),  # 左后轮
        ]
        
        for pos in wheel_positions:
            # 创建车轮
            wheel = pv.Cylinder(
                center=pos,
                radius=self.wheel_radius,
                height=self.wheel_width,
                direction=(0, 1, 0)  # 车轮轴向
            )
            wheels.append(wheel)
        
        return wheels
    
    def get_rover_parameters(self):
        """
        获取月球车参数
        
        Returns:
            dict: 月球车参数
        """
        return {
            'length': self.length,
            'width': self.width,
            'height': self.height,
            'wheel_radius': self.wheel_radius,
            'wheel_width': self.wheel_width,
            'wheel_base': self.wheel_base,
            'wheel_track': self.wheel_track
        }

# 导出类
__all__ = ['RoverModel']

#!/usr/bin/env python3
"""
轨迹生成器模块
提供缓动轨迹生成功能
"""

import numpy as np


class TrajectoryGenerator:
    """轨迹生成器类
    
    提供平滑轨迹生成功能，支持缓动效果
    """
    
    def ease_in_out_cubic(self, t):
        """
        三次缓动函数 (Cubic Ease-In-Out)
        
        Args:
            t: 当前进度 (0.0 ~ 1.0)
            
        Returns:
            缓动后的进度 (0.0 ~ 1.0)
        """
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def generate_smooth_straight_line(self, start_pos, end_pos, duration=10.0, fps=30, max_velocity=0.5):
        """
        生成平滑直线运动轨迹 (无抖动，带缓动)
        
        Args:
            start_pos (tuple): 起点坐标 (x, y) -> (100, 100)
            end_pos (tuple): 终点坐标 (x, y) -> (100, 500)
            duration (float): 运动总时长 (秒)
            fps (int): 帧率 (每秒采样点数)
            max_velocity (float): 最大速度 (m/s)，默认0.5 m/s（月球车实际速度）
            
        Returns:
            np.array: 轨迹点数组，形状 [[x0, y0], [x1, y1], ...]
        """
        # 1. 计算总帧数
        total_frames = int(duration * fps)
        
        # 2. 生成时间进度数组 [0.0, ..., 1.0]
        # 使用 linspace 确保起点和终点严格精确，无浮点数漂移
        times = np.linspace(0, 1, total_frames)
        
        # 3. 计算轨迹
        trajectory = []
        
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        
        # 计算轨迹总长度
        path_length = np.sqrt(delta_x**2 + delta_y**2)
        
        # 调整duration以匹配最大速度
        # 计算理论最小时间
        min_duration = path_length / max_velocity
        if duration < min_duration:
            duration = min_duration
            total_frames = int(duration * fps)
            times = np.linspace(0, 1, total_frames)
        
        for t in times:
            # 应用缓动函数，获取当前"非线性"进度
            eased_t = self.ease_in_out_cubic(t)
            
            # 线性插值计算坐标 (Lerp)
            current_x = start_x + delta_x * eased_t
            current_y = start_y + delta_y * eased_t
            
            trajectory.append([current_x, current_y])
            
        # 4. 确保起点和终点精确
        trajectory = np.array(trajectory)
        trajectory[0] = start_pos
        trajectory[-1] = end_pos
        
        return trajectory

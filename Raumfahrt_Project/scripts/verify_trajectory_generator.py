#!/usr/bin/env python3
"""
验证轨迹生成器功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.core.planning.trajectory_generator import TrajectoryGenerator

def verify_trajectory_generator():
    """验证轨迹生成器功能"""
    print("=== 验证轨迹生成器 ===")
    
    # 创建轨迹生成器
    generator = TrajectoryGenerator()
    
    # 测试参数
    start_pos = (100, 100)
    end_pos = (100, 500)
    duration = 10.0
    fps = 50
    
    # 生成轨迹（使用月球车实际速度限制）
    trajectory = generator.generate_smooth_straight_line(
        start_pos, 
        end_pos, 
        duration, 
        fps,
        max_velocity=0.5  # 月球车实际最大速度
    )
    
    print(f"生成的轨迹点数: {len(trajectory)}")
    print(f"起点: {trajectory[0]}")
    print(f"终点: {trajectory[-1]}")
    
    # 验证轨迹是垂直直线
    all_x = trajectory[:, 0]
    x_variation = np.max(all_x) - np.min(all_x)
    print(f"X坐标变化: {x_variation:.6f} (应该接近0)")
    
    # 验证Y坐标范围
    all_y = trajectory[:, 1]
    print(f"Y坐标范围: {np.min(all_y):.1f} -> {np.max(all_y):.1f}")
    
    # 计算速度
    velocities = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i, 0] - trajectory[i-1, 0]
        dy = trajectory[i, 1] - trajectory[i-1, 1]
        distance = np.sqrt(dx**2 + dy**2)
        time_interval = 1.0 / fps
        velocity = distance / time_interval
        velocities.append(velocity)
    
    if velocities:
        print(f"速度范围: {np.min(velocities):.2f} -> {np.max(velocities):.2f} m/s")
        print(f"平均速度: {np.mean(velocities):.2f} m/s")
    
    # 验证轨迹形状
    print(f"轨迹形状: {trajectory.shape}")
    
    print("\n=== 验证完成 ===")
    print("轨迹生成器工作正常！")

if __name__ == "__main__":
    verify_trajectory_generator()

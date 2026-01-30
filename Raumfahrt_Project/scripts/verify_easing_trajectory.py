#!/usr/bin/env python3
"""
测试缓动轨迹生成器
验证从(100, 100)到(100, 500)的平滑垂直直线轨迹生成
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def ease_in_out_cubic(t):
    """
    三次缓动函数 (Cubic Ease-In-Out)
    t: 当前进度 (0.0 ~ 1.0)
    返回: 缓动后的进度 (0.0 ~ 1.0)
    """
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def generate_smooth_straight_line(start_pos, end_pos, duration=10.0, fps=30, easing_func=ease_in_out_cubic):
    """
    生成平滑直线运动轨迹 (无抖动，带缓动)
    
    Args:
        start_pos (tuple): 起点坐标 (x, y) -> (100, 100)
        end_pos (tuple): 终点坐标 (x, y) -> (100, 500)
        duration (float): 运动总时长 (秒)
        fps (int): 帧率 (每秒采样点数)
        easing_func (func): 缓动函数
        
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
    
    for t in times:
        # 应用缓动函数，获取当前"非线性"进度
        eased_t = easing_func(t)
        
        # 线性插值计算坐标 (Lerp)
        current_x = start_x + delta_x * eased_t
        current_y = start_y + delta_y * eased_t
        
        trajectory.append([current_x, current_y])
        
    return np.array(trajectory)


def run_trajectory_generation():
    """
    测试轨迹生成功能
    """
    print("=== 测试缓动轨迹生成器 ===")
    
    # 配置参数
    START = (100, 100)
    END = (100, 500)
    DURATION = 10.0  # 10秒走完
    FPS = 50        # 50Hz采样
    
    # 生成轨迹
    smooth_path = generate_smooth_straight_line(START, END, DURATION, FPS)
    
    print(f"生成的轨迹点数: {len(smooth_path)}")
    print(f"起点: {smooth_path[0]}")
    print(f"终点: {smooth_path[-1]}")
    print(f"中间点 (5s): {smooth_path[len(smooth_path)//2]}")  # 应该是 (100, 300)
    
    # 验证轨迹是否为垂直直线
    all_x = smooth_path[:, 0]
    x_variation = np.max(all_x) - np.min(all_x)
    print(f"X坐标变化: {x_variation:.6f} (应该接近0)")
    
    # 验证轨迹是否从起点到终点
    start_error = np.linalg.norm(smooth_path[0] - np.array(START))
    end_error = np.linalg.norm(smooth_path[-1] - np.array(END))
    print(f"起点误差: {start_error:.6f} (应该接近0)")
    print(f"终点误差: {end_error:.6f} (应该接近0)")
    
    # 验证中间点是否正确
    middle_idx = len(smooth_path) // 2
    middle_point = smooth_path[middle_idx]
    expected_middle = (START[0], (START[1] + END[1]) / 2)
    middle_error = np.linalg.norm(middle_point - np.array(expected_middle))
    print(f"中间点误差: {middle_error:.6f} (应该接近0)")
    
    # 生成可视化
    visualize_trajectory(smooth_path, START, END, DURATION)
    
    return smooth_path


def visualize_trajectory(trajectory, start, end, duration):
    """
    可视化轨迹
    """
    # 创建可视化目录
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 轨迹图
    plt.figure(figsize=(10, 6))
    
    # 绘制轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='缓动轨迹')
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], color='green', marker='o', s=100, label='起点')
    plt.scatter(end[0], end[1], color='red', marker='*', s=100, label='终点')
    
    # 绘制中间点
    middle_idx = len(trajectory) // 2
    middle_point = trajectory[middle_idx]
    plt.scatter(middle_point[0], middle_point[1], color='orange', marker='s', s=50, label='中间点')
    
    plt.xlabel('X 位置 (m)')
    plt.ylabel('Y 位置 (m)')
    plt.title(f'平滑垂直直线轨迹 (100,100) -> (100,500)\n持续时间: {duration}秒')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存轨迹图
    trajectory_file = os.path.join(viz_dir, 'easing_trajectory.png')
    plt.savefig(trajectory_file, dpi=300)
    plt.close()
    print(f"轨迹图已保存到: {trajectory_file}")
    
    # 速度曲线图
    plt.figure(figsize=(10, 6))
    
    # 计算速度
    velocities = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i, 0] - trajectory[i-1, 0]
        dy = trajectory[i, 1] - trajectory[i-1, 1]
        distance = np.sqrt(dx**2 + dy**2)
        # 假设时间间隔为 1/FPS
        time_interval = 1.0 / 50  # FPS=50
        velocity = distance / time_interval
        velocities.append(velocity)
    
    # 生成时间轴
    time = np.linspace(0, duration, len(velocities))
    
    # 绘制速度曲线
    plt.plot(time, velocities, 'r-', linewidth=2, label='速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.title('速度曲线 (缓动效果)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存速度曲线图
    velocity_file = os.path.join(viz_dir, 'easing_velocity.png')
    plt.savefig(velocity_file, dpi=300)
    plt.close()
    print(f"速度曲线图已保存到: {velocity_file}")


def main():
    """
    主函数
    """
    try:
        trajectory = run_trajectory_generation()
        print("\n=== 测试完成 ===")
        print("轨迹生成成功！")
        return trajectory
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
运行缓动轨迹仿真脚本
使用缓动轨迹生成器生成平滑的垂直直线轨迹，并运行月球车仿真
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.dynamics.rover_dynamics import LunarRoverDynamics
from src.environment.modeling import EnvironmentModeling
from src.core.planning.trajectory_generator import TrajectoryGenerator


# 创建轨迹生成器实例
generator = TrajectoryGenerator()


def load_path_data():
    """
    加载路径数据，使用缓动轨迹生成器生成标准测试轨迹
    """
    print("=== 生成缓动测试轨迹 ===")
    
    # 配置参数
    START = (100, 100)
    END = (100, 500)
    DURATION = 100.0  # 100秒走完，模拟月球车真实速度
    FPS = 50          # 50Hz采样
    MAX_VELOCITY = 0.5  # 月球车最大速度
    
    # 生成轨迹（使用新的轨迹生成器）
    smooth_path = generator.generate_smooth_straight_line(
        START, 
        END, 
        DURATION, 
        FPS,
        max_velocity=MAX_VELOCITY
    )
    
    print(f"生成的轨迹点数: {len(smooth_path)}")
    print(f"起点: {smooth_path[0]}")
    print(f"终点: {smooth_path[-1]}")
    print(f"中间点: {smooth_path[len(smooth_path)//2]}")
    
    # 计算轨迹长度
    path_length = np.sqrt((END[0] - START[0])**2 + (END[1] - START[1])**2)
    print(f"轨迹长度: {path_length:.2f} 米")
    print(f"平均速度: {path_length / DURATION:.2f} 米/秒")
    
    return smooth_path, DURATION, FPS


def run_simulation(path_data, duration, fps):
    """
    运行月球车仿真
    """
    print("\n=== 运行月球车仿真 ===")
    
    # 初始化环境模型
    env_model = EnvironmentModeling()
    print("环境模型初始化完成")
    
    # 初始化月球车动力学模型
    rover = LunarRoverDynamics(env_model=env_model)
    print("动力学模型初始化完成")
    
    # 仿真参数
    dt = 1.0 / fps  # 时间步长
    steps = len(path_data)
    
    # 存储数据
    simulation_data = {
        'time': [],
        'position': [],
        'velocity': [],
        'energy': [],
        'slip_ratio': [],
        'sinkage': [],
        'target_position': []
    }
    
    # 重置状态
    initial_position = [path_data[0][0], path_data[0][1], 0.0]
    rover.reset(initial_position)
    print(f"初始位置: {initial_position}")
    
    print("开始仿真...")
    
    # 运行仿真
    for step in range(steps):
        t = step * dt
        
        # 获取目标位置
        target_pos = path_data[step]
        target_position_3d = [target_pos[0], target_pos[1], 0.0]
        
        # 获取当前状态
        current_state = rover.get_state()
        current_pos = current_state['position'][:2]
        
        # 计算控制命令
        # 简单的比例控制
        error = np.array(target_pos) - np.array(current_pos)
        error_norm = np.linalg.norm(error)
        
        if error_norm > 0:
            direction = error / error_norm
        else:
            direction = np.array([0, 0])
        
        # 计算速度命令
        max_velocity = 0.5  # 最大速度 0.5 m/s
        velocity = min(max_velocity, error_norm * 0.5)  # 比例控制
        
        # 生成车轮速度命令
        wheel_commands = np.array([velocity, velocity, velocity, velocity, velocity, velocity])
        
        # 更新状态
        state = rover.step(wheel_commands, dt)
        
        # 存储数据
        simulation_data['time'].append(t)
        simulation_data['position'].append(state['position'][:2])
        simulation_data['velocity'].append(np.linalg.norm(state['velocity'][:2]))
        simulation_data['energy'].append(state['energy_consumed'])
        simulation_data['slip_ratio'].append(np.mean(state['slip_ratios']))
        simulation_data['sinkage'].append(np.mean(state['sinkages']))
        simulation_data['target_position'].append(target_pos)
        
        # 打印进度
        if step % 500 == 0:
            print(f"进度: {step}/{steps} ({t:.1f}s)")
            print(f"  当前位置: {state['position'][:2]}")
            print(f"  目标位置: {target_pos}")
            print(f"  速度: {np.linalg.norm(state['velocity'][:2]):.2f} m/s")
    
    print("仿真完成！")
    return simulation_data


def save_results(simulation_data, path_data):
    """
    保存仿真结果
    """
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存仿真数据
    simulation_file = os.path.join(results_dir, 'easing_simulation_results.npz')
    np.savez(
        simulation_file,
        time=np.array(simulation_data['time']),
        position=np.array(simulation_data['position']),
        velocity=np.array(simulation_data['velocity']),
        energy=np.array(simulation_data['energy']),
        slip_ratio=np.array(simulation_data['slip_ratio']),
        sinkage=np.array(simulation_data['sinkage']),
        target_position=np.array(simulation_data['target_position']),
        reference_path=np.array(path_data)
    )
    print(f"仿真结果已保存到: {simulation_file}")


def visualize_results(simulation_data, path_data):
    """
    可视化仿真结果
    """
    import matplotlib.pyplot as plt
    
    # 创建可视化目录
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 1. 轨迹对比
    plt.figure(figsize=(12, 8))
    
    # 绘制参考轨迹
    reference_path = np.array(path_data)
    plt.plot(reference_path[:, 0], reference_path[:, 1], 'r--', linewidth=2, label='参考轨迹')
    
    # 绘制实际轨迹
    actual_position = np.array(simulation_data['position'])
    plt.plot(actual_position[:, 0], actual_position[:, 1], 'b-', linewidth=2, label='实际轨迹')
    
    # 绘制起点和终点
    plt.scatter(reference_path[0, 0], reference_path[0, 1], color='green', marker='o', s=100, label='起点')
    plt.scatter(reference_path[-1, 0], reference_path[-1, 1], color='red', marker='*', s=100, label='终点')
    
    plt.xlabel('X 位置 (m)')
    plt.ylabel('Y 位置 (m)')
    plt.title('月球车轨迹对比 (缓动测试)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    trajectory_file = os.path.join(viz_dir, 'easing_trajectory_comparison.png')
    plt.savefig(trajectory_file, dpi=300)
    plt.close()
    print(f"轨迹对比图已保存到: {trajectory_file}")
    
    # 2. 速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_data['time'], simulation_data['velocity'], 'r-', linewidth=2, label='实际速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.title('月球车速度曲线 (缓动测试)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    velocity_file = os.path.join(viz_dir, 'easing_velocity_profile.png')
    plt.savefig(velocity_file, dpi=300)
    plt.close()
    print(f"速度曲线图已保存到: {velocity_file}")
    
    # 3. 位置误差
    plt.figure(figsize=(12, 6))
    target_positions = np.array(simulation_data['target_position'])
    actual_positions = np.array(simulation_data['position'])
    position_error = np.linalg.norm(target_positions - actual_positions, axis=1)
    
    plt.plot(simulation_data['time'], position_error, 'b-', linewidth=2, label='位置误差')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置误差 (m)')
    plt.title('月球车位置误差 (缓动测试)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    error_file = os.path.join(viz_dir, 'easing_position_error.png')
    plt.savefig(error_file, dpi=300)
    plt.close()
    print(f"位置误差图已保存到: {error_file}")


def main():
    """
    主函数
    """
    try:
        # 加载路径数据
        path_data, duration, fps = load_path_data()
        
        # 运行仿真
        simulation_data = run_simulation(path_data, duration, fps)
        
        # 保存结果
        save_results(simulation_data, path_data)
        
        # 可视化结果
        visualize_results(simulation_data, path_data)
        
        print("\n=== 仿真完成 ===")
        print("所有任务已完成！")
        
    except Exception as e:
        print(f"仿真过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

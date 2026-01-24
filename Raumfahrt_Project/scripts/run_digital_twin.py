#!/usr/bin/env python3
"""
运行数字孪生仿真脚本
启动月球车动力学孪生仿真，对比固定参数和自适应参数的性能
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.modeling import EnvironmentModeling
from src.dynamics.rover_dynamics import LunarRoverDynamics
from src.dynamics.estimator import ParameterEstimator


def run_digital_twin_simulation():
    """
    运行数字孪生仿真
    """
    print("=== 月球车数字孪生仿真 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化环境模型
    env_model = EnvironmentModeling()
    print("环境模型初始化完成")
    
    # 初始化月球车动力学模型（固定参数）
    fixed_rover = LunarRoverDynamics(env_model=env_model)
    print("固定参数动力学模型初始化完成")
    
    # 初始化月球车动力学模型（自适应参数）
    adaptive_rover = LunarRoverDynamics(env_model=env_model)
    print("自适应参数动力学模型初始化完成")
    
    # 初始化参数估计器
    estimator = ParameterEstimator()
    print("参数估计器初始化完成")
    
    # 仿真参数
    simulation_time = 100.0  # 仿真时间 (s)
    dt = 0.1  # 时间步长 (s)
    steps = int(simulation_time / dt)
    
    # 控制命令
    wheel_commands = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # 六个车轮的速度命令
    
    # 存储数据
    fixed_data = {
        'time': [],
        'position': [],
        'velocity': [],
        'energy': [],
        'slip_ratio': [],
        'sinkage': [],
    }
    
    adaptive_data = {
        'time': [],
        'position': [],
        'velocity': [],
        'energy': [],
        'slip_ratio': [],
        'sinkage': [],
        'estimated_params': [],
        'confidence': [],
    }
    
    # 重置状态
    start_position = [0.0, 0.0, 0.0]
    fixed_rover.reset(start_position)
    adaptive_rover.reset(start_position)
    
    print("开始仿真...")
    
    # 运行仿真
    for step in range(steps):
        t = step * dt
        
        # 运行固定参数模型
        fixed_state = fixed_rover.step(wheel_commands, dt)
        
        # 运行自适应参数模型
        adaptive_state = adaptive_rover.step(wheel_commands, dt)
        
        # 存储基本数据
        fixed_data['time'].append(t)
        fixed_data['position'].append(fixed_state['position'][:2])
        fixed_data['velocity'].append(np.linalg.norm(fixed_state['velocity'][:2]))
        fixed_data['energy'].append(fixed_state['energy_consumed'])
        fixed_data['slip_ratio'].append(np.mean(fixed_state['slip_ratios']))
        fixed_data['sinkage'].append(np.mean(fixed_state['sinkages']))
        
        adaptive_data['time'].append(t)
        adaptive_data['position'].append(adaptive_state['position'][:2])
        adaptive_data['velocity'].append(np.linalg.norm(adaptive_state['velocity'][:2]))
        adaptive_data['energy'].append(adaptive_state['energy_consumed'])
        adaptive_data['slip_ratio'].append(np.mean(adaptive_state['slip_ratios']))
        adaptive_data['sinkage'].append(np.mean(adaptive_state['sinkages']))
        
        # 估计参数（使用简化的传感器数据）
        if step % 5 == 0:  # 每5步估计一次参数
            sensor_data = {
                'measured_traction': adaptive_state['total_traction'],
                'predicted_traction': fixed_state['total_traction'],
                'slip_ratio': np.mean(adaptive_state['slip_ratios']),
                'sinkage': np.mean(adaptive_state['sinkages']),
                'normal_load': adaptive_rover.params['mass'] * adaptive_rover.params['lunar_gravity'] / 6,
            }
            estimated_params = estimator.estimate_all_parameters(sensor_data)
            confidence = estimator.get_confidence()
            
            # 存储估计的参数和置信度
            adaptive_data['estimated_params'].append(estimator.get_params())
            adaptive_data['confidence'].append(estimator.get_confidence())
            
            # 更新自适应模型的参数
            # 这里简化处理，实际应该更新到具体的参数
        else:
            # 非估计步骤，添加空值
            adaptive_data['estimated_params'].append(None)
            adaptive_data['confidence'].append(None)
        
        # 打印进度
        if step % 100 == 0:
            print(f"进度: {step}/{steps} ({t:.1f}s)")
    
    print("仿真完成！")
    
    # 保存结果
    save_results(fixed_data, adaptive_data)
    
    # 生成可视化
    generate_visualizations(fixed_data, adaptive_data)
    
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=== 仿真完成 ===")


def save_results(fixed_data, adaptive_data):
    """
    保存仿真结果
    """
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存固定参数结果
    fixed_file = os.path.join(results_dir, 'fixed_parameters_results.npz')
    np.savez(
        fixed_file,
        time=np.array(fixed_data['time']),
        position=np.array(fixed_data['position']),
        velocity=np.array(fixed_data['velocity']),
        energy=np.array(fixed_data['energy']),
        slip_ratio=np.array(fixed_data['slip_ratio']),
        sinkage=np.array(fixed_data['sinkage'])
    )
    print(f"固定参数结果已保存到: {fixed_file}")
    
    # 保存自适应参数结果
    adaptive_file = os.path.join(results_dir, 'adaptive_parameters_results.npz')
    np.savez(
        adaptive_file,
        time=np.array(adaptive_data['time']),
        position=np.array(adaptive_data['position']),
        velocity=np.array(adaptive_data['velocity']),
        energy=np.array(adaptive_data['energy']),
        slip_ratio=np.array(adaptive_data['slip_ratio']),
        sinkage=np.array(adaptive_data['sinkage'])
    )
    print(f"自适应参数结果已保存到: {adaptive_file}")


def generate_visualizations(fixed_data, adaptive_data):
    """
    生成可视化图表
    """
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建可视化目录
    viz_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 轨迹对比
    plt.figure(figsize=(10, 8))
    fixed_pos = np.array(fixed_data['position'])
    adaptive_pos = np.array(adaptive_data['position'])
    plt.plot(fixed_pos[:, 0], fixed_pos[:, 1], label='固定参数', linewidth=2)
    plt.plot(adaptive_pos[:, 0], adaptive_pos[:, 1], label='自适应参数', linewidth=2)
    plt.scatter(0, 0, color='green', marker='o', s=100, label='起点')
    plt.scatter(fixed_pos[-1, 0], fixed_pos[-1, 1], color='blue', marker='*', s=100, label='固定参数终点')
    plt.scatter(adaptive_pos[-1, 0], adaptive_pos[-1, 1], color='red', marker='*', s=100, label='自适应参数终点')
    plt.xlabel('X 位置 (m)')
    plt.ylabel('Y 位置 (m)')
    plt.title('月球车轨迹对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    trajectory_file = os.path.join(viz_dir, 'trajectory_comparison.png')
    plt.savefig(trajectory_file, dpi=300)
    plt.close()
    print(f"轨迹对比图已保存到: {trajectory_file}")
    
    # 2. 速度对比
    plt.figure(figsize=(10, 6))
    plt.plot(fixed_data['time'], fixed_data['velocity'], label='固定参数', linewidth=2)
    plt.plot(adaptive_data['time'], adaptive_data['velocity'], label='自适应参数', linewidth=2)
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.title('月球车速度对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    velocity_file = os.path.join(viz_dir, 'velocity_comparison.png')
    plt.savefig(velocity_file, dpi=300)
    plt.close()
    print(f"速度对比图已保存到: {velocity_file}")
    
    # 3. 能量消耗对比
    plt.figure(figsize=(10, 6))
    plt.plot(fixed_data['time'], fixed_data['energy'], label='固定参数', linewidth=2)
    plt.plot(adaptive_data['time'], adaptive_data['energy'], label='自适应参数', linewidth=2)
    plt.xlabel('时间 (s)')
    plt.ylabel('能量消耗 (J)')
    plt.title('月球车能量消耗对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    energy_file = os.path.join(viz_dir, 'energy_comparison.png')
    plt.savefig(energy_file, dpi=300)
    plt.close()
    print(f"能量消耗对比图已保存到: {energy_file}")
    
    # 4. 滑移率对比
    plt.figure(figsize=(10, 6))
    plt.plot(fixed_data['time'], fixed_data['slip_ratio'], label='固定参数', linewidth=2)
    plt.plot(adaptive_data['time'], adaptive_data['slip_ratio'], label='自适应参数', linewidth=2)
    plt.xlabel('时间 (s)')
    plt.ylabel('滑移率')
    plt.title('月球车滑移率对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    slip_file = os.path.join(viz_dir, 'slip_ratio_comparison.png')
    plt.savefig(slip_file, dpi=300)
    plt.close()
    print(f"滑移率对比图已保存到: {slip_file}")
    
    # 5. 参数估计收敛
    if adaptive_data['estimated_params']:
        # 过滤非空的参数估计值
        valid_params = []
        valid_times = []
        valid_confidence = []
        
        for t, p, c in zip(adaptive_data['time'], adaptive_data['estimated_params'], adaptive_data['confidence']):
            if p is not None:
                valid_times.append(t)
                valid_params.append(p)
                valid_confidence.append(c)
        
        if valid_params:
            plt.figure(figsize=(10, 6))
            mu_values = [p['mu'] for p in valid_params]
            plt.plot(valid_times, mu_values, label='摩擦系数 μ', linewidth=2)
            plt.plot(valid_times, valid_confidence, label='置信度', linewidth=2, linestyle='--')
            plt.xlabel('时间 (s)')
            plt.ylabel('值')
            plt.title('参数估计收敛分析')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            params_file = os.path.join(viz_dir, 'parameter_convergence.png')
            plt.savefig(params_file, dpi=300)
            plt.close()
            print(f"参数估计收敛图已保存到: {params_file}")


def main():
    """
    主函数
    """
    try:
        run_digital_twin_simulation()
    except Exception as e:
        print(f"仿真过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

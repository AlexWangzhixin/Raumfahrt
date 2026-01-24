#!/usr/bin/env python3
"""
生成预测轨迹与实际轨迹对比数据
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.environment.terramechanics import Terramechanics, ParameterEstimator
from src.models.dynamics.lunar_rover_dynamics import LunarRoverDynamics

class TrajectoryGenerator:
    """
    轨迹生成器类，用于生成预测轨迹与实际轨迹对比数据
    """
    
    def __init__(self):
        """
        初始化轨迹生成器
        """
        # 初始化地面力学模型
        self.terramechanics = Terramechanics()
        
        # 初始化参数估计器
        self.parameter_estimator = ParameterEstimator()
        
        # 初始化动力学模型
        self.dynamics_model = LunarRoverDynamics()
        
        # 结果数据
        self.results = {
            'predicted_trajectory': [],
            'actual_trajectory': [],
            'predicted_energy': [],
            'actual_energy': [],
            'mu_estimates': [],
            'slip_ratios': [],
            'sinkages': [],
        }
        
        print("轨迹生成器初始化完成")
    
    def generate_trajectories(self, start_position, goal_position, steps=100, dt=0.1):
        """
        生成预测轨迹与实际轨迹
        
        Args:
            start_position: 起始位置 [x, y, z]
            goal_position: 目标位置 [x, y, z]
            steps: 仿真步数
            dt: 时间步长
        
        Returns:
            results: 轨迹对比结果
        """
        print(f"开始生成轨迹: 起点={start_position}, 终点={goal_position}")
        
        # 重置模型
        self.dynamics_model.reset(start_position)
        
        # 计算目标方向
        goal_vector = np.array(goal_position[:2]) - np.array(start_position[:2])
        goal_direction = goal_vector / np.linalg.norm(goal_vector) if np.linalg.norm(goal_vector) > 0 else np.array([1, 0])
        
        # 生成预测轨迹（使用估计的参数）
        predicted_trajectory = [start_position[:2]]
        predicted_energy = []
        mu_estimates = []
        slip_ratios = []
        sinkages = []
        
        # 生成实际轨迹（使用不同的参数）
        actual_trajectory = [start_position[:2]]
        actual_energy = []
        
        # 初始化位置
        current_position = np.array(start_position[:2])
        actual_current_position = np.array(start_position[:2])
        
        # 月球车参数
        mass = 140.0  # kg
        wheel_radius = 0.25  # m
        wheel_width = 0.25  # m
        
        # 目标速度
        max_velocity = 0.3  # m/s
        
        for step in range(steps):
            # 计算到目标的距离
            distance_to_goal = np.linalg.norm(np.array(goal_position[:2]) - current_position)
            if distance_to_goal < 0.5:
                break
            
            # 计算当前方向
            current_direction = goal_vector / np.linalg.norm(goal_vector) if np.linalg.norm(goal_vector) > 0 else np.array([1, 0])
            
            # 预测轨迹计算
            # 计算地形类型（简化为随机生成）
            terrain_type = np.random.choice(['loose_soil', 'firm_soil', 'rock'], p=[0.3, 0.6, 0.1])
            soil_params = self.terramechanics.calculate_soil_parameters(terrain_type)
            
            # 更新地面力学模型参数
            self.terramechanics.update_parameters(soil_params)
            
            # 计算地形可通行性
            terrain_features = {'soil_type': terrain_type}
            traversability = self.terramechanics.calculate_traversability(terrain_features)
            
            # 根据可通行性调整速度
            velocity = max_velocity * traversability
            
            # 计算法向载荷
            normal_load = mass * self.terramechanics.params['gamma']
            
            # 计算沉陷量
            sinkage = self.terramechanics.calculate_sinkage(normal_load)
            sinkages.append(sinkage)
            
            # 计算滚动阻力
            rolling_resistance = self.terramechanics.calculate_rolling_resistance(normal_load)
            
            # 计算滑移率
            slip_ratio = 0.05 * (1 - traversability)
            slip_ratios.append(slip_ratio)
            
            # 计算牵引力
            traction = rolling_resistance
            
            # 计算功率消耗
            power = self.terramechanics.calculate_power_consumption(traction, velocity, rolling_resistance)
            
            # 计算能量消耗
            energy = power * dt
            predicted_energy.append(energy)
            
            # 在线估计摩擦系数
            measured_traction = traction
            predicted_traction = traction
            mu_estimate = self.parameter_estimator.estimate_mu(measured_traction, predicted_traction, slip_ratio)
            mu_estimates.append(mu_estimate)
            
            # 更新地面力学模型的摩擦系数
            self.terramechanics.update_parameters({'mu': mu_estimate})
            
            # 更新预测位置
            current_position += current_direction * velocity * dt
            predicted_trajectory.append(current_position.copy())
            
            # 实际轨迹计算（使用不同的摩擦系数）
            # 模拟实际情况，摩擦系数与估计值有差异
            actual_mu = mu_estimate * (0.8 + 0.4 * np.random.rand())
            actual_velocity = velocity * (0.9 + 0.2 * np.random.rand())
            
            # 更新实际位置
            actual_current_position += current_direction * actual_velocity * dt
            actual_trajectory.append(actual_current_position.copy())
            
            # 计算实际能量消耗
            actual_power = self.terramechanics.calculate_power_consumption(traction, actual_velocity, rolling_resistance)
            actual_energy.append(actual_power * dt)
            
            # 更新目标向量
            goal_vector = np.array(goal_position[:2]) - current_position
            
            if (step + 1) % 10 == 0:
                print(f"仿真步数 {step+1}/{steps} 完成")
        
        # 保存结果
        self.results = {
            'predicted_trajectory': predicted_trajectory,
            'actual_trajectory': actual_trajectory,
            'predicted_energy': predicted_energy,
            'actual_energy': actual_energy,
            'mu_estimates': mu_estimates,
            'slip_ratios': slip_ratios,
            'sinkages': sinkages,
            'total_predicted_energy': sum(predicted_energy),
            'total_actual_energy': sum(actual_energy),
            'trajectory_error': np.linalg.norm(np.array(predicted_trajectory[-1]) - np.array(actual_trajectory[-1])),
        }
        
        print(f"轨迹生成完成，轨迹误差: {self.results['trajectory_error']:.4f}m")
        print(f"预测总能量消耗: {self.results['total_predicted_energy']:.4f}J")
        print(f"实际总能量消耗: {self.results['total_actual_energy']:.4f}J")
        
        return self.results
    
    def visualize_trajectories(self, output_dir='data/visualizations'):
        """
        可视化轨迹对比
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取轨迹数据
        predicted_trajectory = np.array(self.results['predicted_trajectory'])
        actual_trajectory = np.array(self.results['actual_trajectory'])
        
        # 绘制轨迹对比
        plt.figure(figsize=(12, 10))
        
        # 绘制预测轨迹
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'b-', linewidth=2, label='预测轨迹')
        plt.scatter(predicted_trajectory[0, 0], predicted_trajectory[0, 1], c='g', marker='o', s=100, label='起点')
        plt.scatter(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], c='r', marker='x', s=100, label='预测终点')
        
        # 绘制实际轨迹
        plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'r--', linewidth=2, label='实际轨迹')
        plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], c='m', marker='s', s=100, label='实际终点')
        
        # 设置图表属性
        plt.xlabel('X坐标 (m)', fontsize=12)
        plt.ylabel('Y坐标 (m)', fontsize=12)
        plt.title('预测轨迹与实际轨迹对比', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 保存轨迹对比图
        trajectory_path = os.path.join(output_dir, 'trajectory_comparison.png')
        plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"轨迹对比图已保存到: {trajectory_path}")
        
        # 绘制摩擦系数估计
        if self.results['mu_estimates']:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(self.results['mu_estimates'])), self.results['mu_estimates'], 'g-', linewidth=2)
            plt.xlabel('时间步', fontsize=12)
            plt.ylabel('摩擦系数估计值', fontsize=12)
            plt.title('摩擦系数估计收敛曲线', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            mu_path = os.path.join(output_dir, 'mu_estimate_convergence.png')
            plt.savefig(mu_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"摩擦系数估计收敛曲线已保存到: {mu_path}")
        
        # 绘制能量消耗对比
        if self.results['predicted_energy'] and self.results['actual_energy']:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(self.results['predicted_energy'])), self.results['predicted_energy'], 'b-', linewidth=2, label='预测能量消耗')
            plt.plot(range(len(self.results['actual_energy'])), self.results['actual_energy'], 'r--', linewidth=2, label='实际能量消耗')
            plt.xlabel('时间步', fontsize=12)
            plt.ylabel('能量消耗 (J)', fontsize=12)
            plt.title('能量消耗对比', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            energy_path = os.path.join(output_dir, 'energy_consumption_comparison.png')
            plt.savefig(energy_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"能量消耗对比图已保存到: {energy_path}")
        
        # 绘制沉陷量和滑移率
        if self.results['sinkages'] and self.results['slip_ratios']:
            plt.figure(figsize=(14, 6))
            
            # 绘制沉陷量
            plt.subplot(1, 2, 1)
            plt.plot(range(len(self.results['sinkages'])), self.results['sinkages'], 'b-', linewidth=2)
            plt.xlabel('时间步', fontsize=12)
            plt.ylabel('沉陷量 (m)', fontsize=12)
            plt.title('车轮沉陷量', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 绘制滑移率
            plt.subplot(1, 2, 2)
            plt.plot(range(len(self.results['slip_ratios'])), self.results['slip_ratios'], 'r-', linewidth=2)
            plt.xlabel('时间步', fontsize=12)
            plt.ylabel('滑移率', fontsize=12)
            plt.title('车轮滑移率', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            terramechanics_path = os.path.join(output_dir, 'terramechanics_data.png')
            plt.tight_layout()
            plt.savefig(terramechanics_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"地面力学数据图已保存到: {terramechanics_path}")
    
    def save_results(self, output_dir='data/visualizations'):
        """
        保存轨迹对比结果
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存轨迹数据
        trajectory_data = {
            'predicted_trajectory': self.results['predicted_trajectory'],
            'actual_trajectory': self.results['actual_trajectory'],
            'predicted_energy': self.results['predicted_energy'],
            'actual_energy': self.results['actual_energy'],
            'mu_estimates': self.results['mu_estimates'],
            'slip_ratios': self.results['slip_ratios'],
            'sinkages': self.results['sinkages'],
            'total_predicted_energy': sum(self.results['predicted_energy']),
            'total_actual_energy': sum(self.results['actual_energy']),
            'trajectory_error': self.results['trajectory_error'],
        }
        
        # 保存为npz文件
        data_path = os.path.join(output_dir, 'trajectory_comparison_data.npz')
        np.savez(data_path, **trajectory_data)
        print(f"轨迹对比数据已保存到: {data_path}")
        
        # 保存为文本文件
        text_path = os.path.join(output_dir, 'trajectory_comparison.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("轨迹对比结果\n")
            f.write("=============\n\n")
            f.write(f"总预测能量消耗: {sum(self.results['predicted_energy']):.4f}J\n")
            f.write(f"总实际能量消耗: {sum(self.results['actual_energy']):.4f}J\n")
            f.write(f"轨迹终点误差: {self.results['trajectory_error']:.4f}m\n")
            f.write(f"平均摩擦系数估计: {np.mean(self.results['mu_estimates']):.4f}\n")
            f.write(f"平均沉陷量: {np.mean(self.results['sinkages']):.4f}m\n")
            f.write(f"平均滑移率: {np.mean(self.results['slip_ratios']):.4f}\n")
        
        print(f"轨迹对比结果已保存到: {text_path}")

if __name__ == "__main__":
    # 创建轨迹生成器
    generator = TrajectoryGenerator()
    
    # 定义起点和终点
    start_position = [0.0, 0.0, 0.0]
    goal_position = [40.0, 40.0, 0.0]
    
    # 生成轨迹
    results = generator.generate_trajectories(start_position, goal_position, steps=200, dt=0.1)
    
    # 可视化轨迹对比
    generator.visualize_trajectories()
    
    # 保存结果
    generator.save_results()
    
    print("\n轨迹对比数据生成完成！")

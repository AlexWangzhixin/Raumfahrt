#!/usr/bin/env python3
"""
结果分析脚本
用于分析仿真结果和训练数据
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from src.core.utils import load_json, load_yaml, ensure_directory
from src.core.visualization import Visualization

class ResultsAnalyzer:
    """
    结果分析类
    用于分析仿真结果和训练数据
    """
    
    def __init__(self):
        """
        初始化分析器
        """
        print("初始化结果分析器...")
        
        # 初始化可视化工具
        self.vis = Visualization()
        
        # 创建分析输出目录
        self.output_dir = os.path.join('outputs', 'analysis')
        ensure_directory(self.output_dir)
        
        print(f"分析输出目录: {self.output_dir}")
    
    def analyze_simulation_results(self, results_path):
        """
        分析仿真结果
        
        Args:
            results_path: 仿真结果文件路径
        """
        print(f"分析仿真结果: {results_path}")
        
        # 加载结果数据
        data = np.load(results_path)
        
        # 提取数据
        adaptive_params = data.get('adaptive_parameters', None)
        fixed_params = data.get('fixed_parameters', None)
        trajectory = data.get('trajectory', None)
        time_steps = data.get('time_steps', None)
        
        # 分析动力学性能
        if adaptive_params is not None and fixed_params is not None:
            self._analyze_dynamics_performance(
                adaptive_params, fixed_params, time_steps
            )
        
        # 分析轨迹
        if trajectory is not None:
            self._analyze_trajectory(trajectory)
    
    def analyze_training_results(self, training_logs_path):
        """
        分析训练结果
        
        Args:
            training_logs_path: 训练日志文件路径
        """
        print(f"分析训练结果: {training_logs_path}")
        
        # 加载训练数据
        data = np.load(training_logs_path)
        
        # 提取数据
        episode_rewards = data.get('episode_rewards', None)
        episode_lengths = data.get('episode_lengths', None)
        q_values = data.get('q_values', None)
        
        # 分析训练性能
        if episode_rewards is not None:
            self._analyze_training_performance(
                episode_rewards, episode_lengths, q_values
            )
    
    def _analyze_dynamics_performance(self, adaptive_params, fixed_params, time_steps):
        """
        分析动力学性能
        
        Args:
            adaptive_params: 自适应参数结果
            fixed_params: 固定参数结果
            time_steps: 时间步
        """
        print("分析动力学性能...")
        
        # 创建图表
        fig, axes = self.vis.create_figure(subplots=(2, 2))
        
        # 绘制速度对比
        if 'velocity' in adaptive_params and 'velocity' in fixed_params:
            ax = axes[0, 0]
            ax.plot(time_steps, adaptive_params['velocity'], 'r-', label='Adaptive')
            ax.plot(time_steps, fixed_params['velocity'], 'b-', label='Fixed')
            ax.set_title('Velocity Comparison')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 绘制加速度对比
        if 'acceleration' in adaptive_params and 'acceleration' in fixed_params:
            ax = axes[0, 1]
            ax.plot(time_steps, adaptive_params['acceleration'], 'r-', label='Adaptive')
            ax.plot(time_steps, fixed_params['acceleration'], 'b-', label='Fixed')
            ax.set_title('Acceleration Comparison')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 绘制能耗对比
        if 'energy' in adaptive_params and 'energy' in fixed_params:
            ax = axes[1, 0]
            ax.plot(time_steps, adaptive_params['energy'], 'r-', label='Adaptive')
            ax.plot(time_steps, fixed_params['energy'], 'b-', label='Fixed')
            ax.set_title('Energy Consumption')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Energy (J)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 绘制滑移率对比
        if 'slip_ratio' in adaptive_params and 'slip_ratio' in fixed_params:
            ax = axes[1, 1]
            ax.plot(time_steps, adaptive_params['slip_ratio'], 'r-', label='Adaptive')
            ax.plot(time_steps, fixed_params['slip_ratio'], 'b-', label='Fixed')
            ax.set_title('Slip Ratio')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Slip Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 调整布局并保存
        plt.tight_layout()
        output_path = os.path.join(
            self.output_dir, 'dynamics_performance_comparison.png'
        )
        self.vis.save_figure(fig, output_path)
        print(f"保存动力学性能对比图到: {output_path}")
    
    def _analyze_trajectory(self, trajectory):
        """
        分析轨迹
        
        Args:
            trajectory: 轨迹数据
        """
        print("分析轨迹...")
        
        # 创建图表
        fig, ax = self.vis.create_figure()
        
        # 绘制轨迹
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        
        # 绘制起点和终点
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
        
        # 设置图表属性
        ax.set_title('Trajectory Analysis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 计算轨迹统计信息
        total_distance = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i, 0] - trajectory[i-1, 0]
            dy = trajectory[i, 1] - trajectory[i-1, 1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        start_end_distance = np.sqrt(
            (trajectory[-1, 0] - trajectory[0, 0])**2 +
            (trajectory[-1, 1] - trajectory[0, 1])**2
        )
        
        efficiency = start_end_distance / total_distance if total_distance > 0 else 0
        
        print(f"轨迹长度: {total_distance:.2f} m")
        print(f"起止距离: {start_end_distance:.2f} m")
        print(f"轨迹效率: {efficiency:.2f}")
        
        # 保存图表
        output_path = os.path.join(
            self.output_dir, 'trajectory_analysis.png'
        )
        self.vis.save_figure(fig, output_path)
        print(f"保存轨迹分析图到: {output_path}")
        
        # 保存轨迹统计信息
        stats_path = os.path.join(
            self.output_dir, 'trajectory_stats.txt'
        )
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("轨迹统计信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"轨迹长度: {total_distance:.2f} m\n")
            f.write(f"起止距离: {start_end_distance:.2f} m\n")
            f.write(f"轨迹效率: {efficiency:.2f}\n")
            f.write(f"轨迹点数量: {len(trajectory)}\n")
            f.write(f"起点坐标: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f})\n")
            f.write(f"终点坐标: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})\n")
        
        print(f"保存轨迹统计信息到: {stats_path}")
    
    def _analyze_training_performance(self, episode_rewards, episode_lengths, q_values):
        """
        分析训练性能
        
        Args:
            episode_rewards:  episode奖励
            episode_lengths: episode长度
            q_values: Q值
        """
        print("分析训练性能...")
        
        # 创建图表
        fig, axes = self.vis.create_figure(subplots=(2, 2))
        
        # 绘制奖励曲线
        ax = axes[0, 0]
        ax.plot(episode_rewards, 'b-')
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        # 绘制移动平均奖励
        ax = axes[0, 1]
        window_size = 100
        if len(episode_rewards) > window_size:
            moving_avg = np.convolve(
                episode_rewards, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            ax.plot(moving_avg, 'r-')
            ax.set_title(f'Moving Average Reward (window={window_size})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Reward')
            ax.grid(True, alpha=0.3)
        
        # 绘制episode长度
        if episode_lengths is not None:
            ax = axes[1, 0]
            ax.plot(episode_lengths, 'g-')
            ax.set_title('Episode Lengths')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Length')
            ax.grid(True, alpha=0.3)
        
        # 绘制Q值
        if q_values is not None:
            ax = axes[1, 1]
            ax.plot(q_values, 'y-')
            ax.set_title('Q Values')
            ax.set_xlabel('Step')
            ax.set_ylabel('Q Value')
            ax.grid(True, alpha=0.3)
        
        # 调整布局并保存
        plt.tight_layout()
        output_path = os.path.join(
            self.output_dir, 'training_performance.png'
        )
        self.vis.save_figure(fig, output_path)
        print(f"保存训练性能图到: {output_path}")
        
        # 计算训练统计信息
        avg_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)
        
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"最大奖励: {max_reward:.2f}")
        print(f"最小奖励: {min_reward:.2f}")
        
        # 保存训练统计信息
        stats_path = os.path.join(
            self.output_dir, 'training_stats.txt'
        )
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("训练统计信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"总Episodes: {len(episode_rewards)}\n")
            f.write(f"平均奖励: {avg_reward:.2f}\n")
            f.write(f"最大奖励: {max_reward:.2f}\n")
            f.write(f"最小奖励: {min_reward:.2f}\n")
            if episode_lengths is not None:
                avg_length = np.mean(episode_lengths)
                f.write(f"平均Episode长度: {avg_length:.2f}\n")
            if q_values is not None:
                avg_q = np.mean(q_values)
                f.write(f"平均Q值: {avg_q:.2f}\n")
        
        print(f"保存训练统计信息到: {stats_path}")
    
    def batch_analyze(self, directory):
        """
        批量分析目录中的结果文件
        
        Args:
            directory: 结果文件目录
        """
        print(f"批量分析目录: {directory}")
        
        # 遍历目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.npz'):
                    file_path = os.path.join(root, file)
                    print(f"分析文件: {file_path}")
                    
                    try:
                        # 尝试加载文件
                        data = np.load(file_path)
                        
                        # 判断文件类型
                        if 'episode_rewards' in data:
                            self.analyze_training_results(file_path)
                        else:
                            self.analyze_simulation_results(file_path)
                    except Exception as e:
                        print(f"分析文件 {file_path} 时出错: {e}")

if __name__ == '__main__':
    analyzer = ResultsAnalyzer()
    
    # 示例用法
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            analyzer.batch_analyze(path)
        elif os.path.isfile(path):
            if 'training' in path.lower():
                analyzer.analyze_training_results(path)
            else:
                analyzer.analyze_simulation_results(path)
    else:
        # 默认分析输出目录
        analyzer.batch_analyze('outputs')

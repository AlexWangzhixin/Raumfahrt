#!/usr/bin/env python3
"""
系统演示脚本
演示完整的月球车导航系统
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.core.utils import ensure_directory, get_timestamp
from src.core.visualization import Visualization
from src.environment.modeling import EnvironmentModeling
from src.dynamics.rover_dynamics import LunarRoverDynamics
from src.planning.global_planner.astar import AStarPlanner
from src.planning.local_planner.agent import D3QNAgent

class SystemDemo:
    """
    系统演示类
    演示完整的月球车导航系统
    """
    
    def __init__(self):
        """
        初始化演示系统
        """
        print("初始化月球车导航系统演示...")
        
        # 创建时间戳
        self.timestamp = get_timestamp()
        
        # 创建输出目录
        self.output_dir = os.path.join(
            'outputs', 'visualizations', f'demo_{self.timestamp}'
        )
        ensure_directory(self.output_dir)
        
        # 初始化可视化工具
        self.vis = Visualization()
        
        # 初始化环境模型
        self.env = EnvironmentModeling()
        
        # 初始化动力学模型
        self.dynamics = LunarRoverDynamics()
        
        # 初始化全局规划器
        self.global_planner = AStarPlanner()
        
        # 初始化局部规划器
        self.local_planner = D3QNAgent(
            use_per=True, 
            use_astar=True
        )
        
        # 配置演示参数
        self.demo_params = {
            'start_position': (0.0, 0.0),
            'goal_position': (100.0, 80.0),
            'simulation_duration': 600,  # 10分钟
            'time_step': 0.1,            # 100ms
            'obstacle_density': 0.1,      # 10% 障碍物密度
        }
        
    def run(self):
        """
        运行演示
        """
        print("开始系统演示...")
        print(f"演示参数: {self.demo_params}")
        print(f"输出目录: {self.output_dir}")
        
        # 步骤1: 生成环境地图
        print("步骤1: 生成环境地图...")
        map_data = self._generate_demo_map()
        
        # 步骤2: 全局路径规划
        print("步骤2: 全局路径规划...")
        global_path = self.global_planner.plan_path(
            self.demo_params['start_position'],
            self.demo_params['goal_position'],
            map_data=map_data
        )
        
        # 步骤3: 动力学仿真
        print("步骤3: 动力学仿真...")
        trajectory, states = self._run_simulation(global_path)
        
        # 步骤4: 可视化结果
        print("步骤4: 可视化结果...")
        self._visualize_results(map_data, global_path, trajectory)
        
        # 步骤5: 生成报告
        print("步骤5: 生成演示报告...")
        self._generate_report(trajectory, states)
        
        print("系统演示完成！")
    
    def _generate_demo_map(self):
        """
        生成演示地图
        
        Returns:
            map_data: 地图数据
        """
        # 生成一个简单的地图
        map_size = (200, 200)
        map_data = np.zeros(map_size)
        
        # 添加一些障碍物
        for i in range(map_size[0]):
            for j in range(map_size[1]):
                if np.random.random() < self.demo_params['obstacle_density']:
                    map_data[i, j] = 1  # 1 表示障碍物
        
        # 确保起点和终点没有障碍物
        start_x, start_y = self.demo_params['start_position']
        goal_x, goal_y = self.demo_params['goal_position']
        map_data[int(start_x), int(start_y)] = 0
        map_data[int(goal_x), int(goal_y)] = 0
        
        return map_data
    
    def _run_simulation(self, global_path):
        """
        运行动力学仿真
        
        Args:
            global_path: 全局路径
        
        Returns:
            trajectory: 实际轨迹
            states: 状态历史
        """
        trajectory = []
        states = []
        
        # 重置动力学模型
        initial_state = {
            'position': np.array([self.demo_params['start_position'][0], self.demo_params['start_position'][1], 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),
        }
        self.dynamics.reset(initial_state)
        
        # 运行仿真
        current_time = 0
        while current_time < self.demo_params['simulation_duration']:
            # 获取当前状态
            state = self.dynamics.get_state()
            states.append(state)
            
            # 记录轨迹
            position = state['position'][:2]
            trajectory.append((position[0], position[1]))
            
            # 计算控制输入
            control = self._compute_control(state, global_path)
            
            # 更新动力学状态
            self.dynamics.update_state(control, self.demo_params['time_step'])
            
            # 更新时间
            current_time += self.demo_params['time_step']
            
            # 检查是否到达目标
            distance_to_goal = np.sqrt(
                (position[0] - self.demo_params['goal_position'][0])**2 +
                (position[1] - self.demo_params['goal_position'][1])**2
            )
            if distance_to_goal < 1.0:  # 1米内视为到达
                print(f"到达目标！距离: {distance_to_goal:.2f}m")
                break
        
        return trajectory, states
    
    def _compute_control(self, state, global_path):
        """
        计算控制输入
        
        Args:
            state: 当前状态
            global_path: 全局路径
        
        Returns:
            control: 控制输入
        """
        # 简单的控制逻辑
        position = state['position'][:2]
        
        # 找到路径上的下一个点
        closest_idx = np.argmin([
            np.sqrt((x - position[0])**2 + (y - position[1])**2)
            for x, y in global_path
        ])
        
        next_idx = min(closest_idx + 5, len(global_path) - 1)
        target_position = global_path[next_idx]
        
        # 计算方向
        direction = np.array(target_position) - np.array(position)
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
        
        # 计算控制量
        control = {
            'linear_velocity': 0.5,  # 0.5 m/s
            'angular_velocity': 0.0,  # 暂时设为0
        }
        
        return control
    
    def _visualize_results(self, map_data, global_path, trajectory):
        """
        可视化结果
        
        Args:
            map_data: 地图数据
            global_path: 全局路径
            trajectory: 实际轨迹
        """
        # 绘制地图和路径
        fig, ax = self.vis.create_figure()
        
        # 绘制地图
        ax.imshow(
            map_data, 
            cmap='gray', 
            origin='lower',
            alpha=0.5
        )
        
        # 绘制全局路径
        global_path = np.array(global_path)
        ax.plot(
            global_path[:, 0], 
            global_path[:, 1], 
            'r--', 
            linewidth=2, 
            label='Global Path'
        )
        
        # 绘制实际轨迹
        trajectory = np.array(trajectory)
        ax.plot(
            trajectory[:, 0], 
            trajectory[:, 1], 
            'b-', 
            linewidth=2, 
            label='Actual Trajectory'
        )
        
        # 绘制起点和终点
        ax.plot(
            self.demo_params['start_position'][0],
            self.demo_params['start_position'][1],
            'go', 
            markersize=10, 
            label='Start'
        )
        ax.plot(
            self.demo_params['goal_position'][0],
            self.demo_params['goal_position'][1],
            'ro', 
            markersize=10, 
            label='Goal'
        )
        
        ax.set_title('月球车导航系统演示')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存结果
        map_plot_path = os.path.join(self.output_dir, 'navigation_demo.png')
        self.vis.save_figure(fig, map_plot_path)
        print(f"保存导航演示图到: {map_plot_path}")
    
    def _generate_report(self, trajectory, states):
        """
        生成演示报告
        
        Args:
            trajectory: 实际轨迹
            states: 状态历史
        """
        # 计算统计信息
        total_distance = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        average_speed = total_distance / len(states) / self.demo_params['time_step']
        
        # 生成报告
        report_path = os.path.join(self.output_dir, 'demo_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("月球车导航系统演示报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"演示时间: {self.timestamp}\n")
            f.write(f"起点位置: {self.demo_params['start_position']}\n")
            f.write(f"目标位置: {self.demo_params['goal_position']}\n")
            f.write(f"总距离: {total_distance:.2f} m\n")
            f.write(f"平均速度: {average_speed:.2f} m/s\n")
            f.write(f"仿真步数: {len(states)}\n")
            f.write(f"仿真时间: {len(states) * self.demo_params['time_step']:.1f} s\n")
            f.write("=" * 50 + "\n")
            f.write("演示完成！\n")
        
        print(f"保存演示报告到: {report_path}")

if __name__ == '__main__':
    demo = SystemDemo()
    demo.run()

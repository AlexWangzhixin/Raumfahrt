#!/usr/bin/env python3
"""
月球车动力学建模模块
基于嫦娥6号数据的月球车动力学模型
"""

import numpy as np

class LunarRoverDynamics:
    """
    月球车动力学模型类
    """
    
    def __init__(self, rover_params=None):
        """
        初始化月球车动力学模型
        
        Args:
            rover_params: 月球车参数
        """
        # 默认参数
        default_params = {
            'mass': 140.0,  # 质量 (kg)
            'wheel_radius': 0.25,  # 车轮半径 (m)
            'wheel_base': 1.5,  # 轴距 (m)
            'track_width': 1.0,  # 轮距 (m)
            'max_wheel_speed': 10.0,  # 最大车轮速度 (rad/s)
            'max_torque': 50.0,  # 最大扭矩 (N·m)
            'drag_coefficient': 0.1,  # 阻力系数
            'rolling_resistance': 0.05,  # 滚动阻力系数
            'lunar_gravity': 1.62,  # 月球重力加速度 (m/s²)
            'inertia_tensor': np.diag([10.0, 10.0, 20.0]),  # 惯性张量
        }
        
        # 合并参数
        self.params = default_params.copy()
        if rover_params:
            self.params.update(rover_params)
        
        # 状态变量
        self.state = {
            'position': np.array([0.0, 0.0, 0.0]),  # 位置 (x, y, z)
            'velocity': np.array([0.0, 0.0, 0.0]),  # 速度 (x, y, z)
            'orientation': np.array([0.0, 0.0, 0.0]),  # 姿态 (roll, pitch, yaw)
            'angular_velocity': np.array([0.0, 0.0, 0.0]),  # 角速度
            'wheel_speeds': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 八个车轮的速度
        }
        
        # 能量消耗
        self.energy_consumed = 0.0
        
        # 接触状态
        self.contact_states = np.ones(8, dtype=bool)  # 八个车轮的接触状态
        
        print("月球车动力学模型初始化完成")
    
    def reset(self, start_position):
        """
        重置动力学模型状态
        
        Args:
            start_position: 起始位置 [x, y, z]
        """
        self.state['position'] = np.array(start_position)
        self.state['velocity'] = np.array([0.0, 0.0, 0.0])
        self.state['orientation'] = np.array([0.0, 0.0, 0.0])
        self.state['angular_velocity'] = np.array([0.0, 0.0, 0.0])
        self.state['wheel_speeds'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.energy_consumed = 0.0
        self.contact_states = np.ones(8, dtype=bool)
        
        print(f"动力学模型重置完成，起始位置: {start_position}")
    
    def step(self, wheel_commands, dt):
        """
        执行动力学仿真一步
        
        Args:
            wheel_commands: 八个车轮的控制命令 (rad/s)
            dt: 时间步长 (s)
        
        Returns:
            state_info: 状态信息
        """
        # 限制车轮速度
        wheel_speeds = np.clip(wheel_commands, -self.params['max_wheel_speed'], self.params['max_wheel_speed'])
        self.state['wheel_speeds'] = wheel_speeds
        
        # 计算车身速度
        # 简化模型：假设左右两侧车轮速度相同
        left_speed = np.mean(wheel_speeds[[0, 2, 4, 6]])  # 左侧车轮
        right_speed = np.mean(wheel_speeds[[1, 3, 5, 7]])  # 右侧车轮
        
        # 计算线速度和角速度
        linear_velocity = (left_speed + right_speed) * self.params['wheel_radius'] / 2.0
        angular_velocity = (right_speed - left_speed) * self.params['wheel_radius'] / self.params['track_width']
        
        # 计算车身速度向量
        yaw = self.state['orientation'][2]
        velocity = np.array([
            linear_velocity * np.cos(yaw),
            linear_velocity * np.sin(yaw),
            0.0  # 假设在平面上运动
        ])
        
        # 更新速度
        self.state['velocity'] = velocity
        self.state['angular_velocity'][2] = angular_velocity
        
        # 更新位置
        self.state['position'] += velocity * dt
        
        # 更新姿态
        self.state['orientation'][2] += angular_velocity * dt
        # 限制偏航角在 [-π, π] 范围内
        self.state['orientation'][2] = np.arctan2(np.sin(self.state['orientation'][2]), np.cos(self.state['orientation'][2]))
        
        # 计算能量消耗
        # 简化模型：能量消耗与车轮速度的平方成正比
        power = np.sum(np.square(wheel_speeds)) * 0.1  # 简化的功率计算
        energy = power * dt
        self.energy_consumed += energy
        
        # 更新接触状态
        # 简化模型：假设所有车轮都与地面接触
        self.contact_states = np.ones(8, dtype=bool)
        
        # 构建状态信息
        state_info = {
            'position': self.state['position'].copy(),
            'velocity': self.state['velocity'].copy(),
            'orientation': self.state['orientation'].copy(),
            'angular_velocity': self.state['angular_velocity'].copy(),
            'wheel_speeds': self.state['wheel_speeds'].copy(),
            'energy_consumed': self.energy_consumed,
            'contact_states': self.contact_states.copy(),
        }
        
        return state_info
    
    def get_state(self):
        """
        获取当前状态
        
        Returns:
            当前状态
        """
        return self.state.copy()
    
    def set_state(self, state):
        """
        设置状态
        
        Args:
            state: 新的状态
        """
        self.state.update(state)
    
    def compute_traversability(self, terrain_features):
        """
        计算地形可通行性
        
        Args:
            terrain_features: 地形特征
        
        Returns:
            traversability: 可通行性分数 [0, 1]
        """
        # 简化的可通行性计算
        traversability = 1.0
        
        # 考虑粗糙度
        if 'roughness' in terrain_features:
            roughness = terrain_features['roughness']
            traversability *= max(0, 1 - roughness * 2)
        
        # 考虑坡度
        if 'slope' in terrain_features:
            slope = terrain_features['slope']
            traversability *= max(0, 1 - slope * 3)
        
        # 考虑曲率
        if 'curvature' in terrain_features:
            curvature = terrain_features['curvature']
            traversability *= max(0, 1 - curvature * 5)
        
        return min(1.0, max(0.0, traversability))
    
    def predict_motion(self, wheel_commands, dt, steps=1):
        """
        预测未来运动
        
        Args:
            wheel_commands: 车轮控制命令
            dt: 时间步长
            steps: 预测步数
        
        Returns:
            predicted_states: 预测的状态序列
        """
        predicted_states = []
        
        # 保存当前状态
        original_state = self.state.copy()
        original_energy = self.energy_consumed
        original_contact = self.contact_states.copy()
        
        try:
            # 执行预测
            for i in range(steps):
                state_info = self.step(wheel_commands, dt)
                predicted_states.append(state_info)
        finally:
            # 恢复原始状态
            self.state = original_state
            self.energy_consumed = original_energy
            self.contact_states = original_contact
        
        return predicted_states
    
    def get_kinematic_model(self):
        """
        获取运动学模型
        
        Returns:
            kinematic_model: 运动学模型函数
        """
        def kinematic_model(v, w, dt):
            """
            运动学模型
            
            Args:
                v: 线速度
                w: 角速度
                dt: 时间步长
            
            Returns:
                dx, dy, dtheta: 位置和姿态变化
            """
            if abs(w) < 1e-6:
                # 直线运动
                dx = v * dt * np.cos(self.state['orientation'][2])
                dy = v * dt * np.sin(self.state['orientation'][2])
                dtheta = 0.0
            else:
                # 圆弧运动
                radius = v / w
                dtheta = w * dt
                dx = radius * (np.sin(self.state['orientation'][2] + dtheta) - np.sin(self.state['orientation'][2]))
                dy = -radius * (np.cos(self.state['orientation'][2] + dtheta) - np.cos(self.state['orientation'][2]))
            
            return dx, dy, dtheta
        
        return kinematic_model
#!/usr/bin/env python3
"""
动力学与感知集成模块
将月球车动力学与感知系统集成
"""

import numpy as np

class DynamicsPerceptionIntegration:
    """
    动力学与感知集成类
    """
    
    def __init__(self, dt=0.1):
        """
        初始化动力学与感知集成模块
        
        Args:
            dt: 时间步长
        """
        self.dt = dt
        
        # 状态估计
        self.state_estimate = {
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.0, 0.0, 0.0]),
        }
        
        # 感知数据缓冲区
        self.perception_buffer = []
        
        # 预测状态
        self.predicted_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.0, 0.0, 0.0]),
        }
        
        # 不确定性
        self.uncertainty = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # [x, y, z, roll, pitch, yaw]的不确定性
        
        print("动力学与感知集成模块初始化完成")
    
    def reset(self, start_position):
        """
        重置集成模块状态
        
        Args:
            start_position: 起始位置
        """
        self.state_estimate['position'] = np.array(start_position)
        self.state_estimate['velocity'] = np.array([0.0, 0.0, 0.0])
        self.state_estimate['orientation'] = np.array([0.0, 0.0, 0.0])
        self.state_estimate['angular_velocity'] = np.array([0.0, 0.0, 0.0])
        
        self.predicted_state['position'] = np.array(start_position)
        self.predicted_state['velocity'] = np.array([0.0, 0.0, 0.0])
        self.predicted_state['orientation'] = np.array([0.0, 0.0, 0.0])
        self.predicted_state['angular_velocity'] = np.array([0.0, 0.0, 0.0])
        
        self.perception_buffer = []
        self.uncertainty = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        
        print(f"动力学与感知集成模块重置完成，起始位置: {start_position}")
    
    def step(self, wheel_commands, dt):
        """
        执行集成模块一步
        
        Args:
            wheel_commands: 车轮控制命令
            dt: 时间步长
        
        Returns:
            integration_state: 集成状态
        """
        # 预测状态
        self._predict_state(wheel_commands, dt)
        
        # 更新不确定性
        self._update_uncertainty()
        
        # 融合感知数据
        if self.perception_buffer:
            self._fuse_perception_data()
        
        # 构建集成状态
        integration_state = {
            'state_estimate': self.state_estimate.copy(),
            'predicted_state': self.predicted_state.copy(),
            'uncertainty': self.uncertainty.copy(),
            'perception_buffer_size': len(self.perception_buffer),
        }
        
        return integration_state
    
    def _predict_state(self, wheel_commands, dt):
        """
        预测状态
        
        Args:
            wheel_commands: 车轮控制命令
            dt: 时间步长
        """
        # 简化的状态预测
        # 基于车轮命令计算速度
        left_speed = np.mean(wheel_commands[[0, 2, 4, 6]])
        right_speed = np.mean(wheel_commands[[1, 3, 5, 7]])
        
        # 假设车轮半径为0.25m
        wheel_radius = 0.25
        # 假设轮距为1.0m
        track_width = 1.0
        
        # 计算线速度和角速度
        linear_velocity = (left_speed + right_speed) * wheel_radius / 2.0
        angular_velocity = (right_speed - left_speed) * wheel_radius / track_width
        
        # 计算速度向量
        yaw = self.state_estimate['orientation'][2]
        velocity = np.array([
            linear_velocity * np.cos(yaw),
            linear_velocity * np.sin(yaw),
            0.0
        ])
        
        # 更新预测状态
        self.predicted_state['position'] = self.state_estimate['position'] + velocity * dt
        self.predicted_state['velocity'] = velocity
        self.predicted_state['orientation'][2] += angular_velocity * dt
        self.predicted_state['angular_velocity'][2] = angular_velocity
        
        # 限制偏航角
        self.predicted_state['orientation'][2] = np.arctan2(
            np.sin(self.predicted_state['orientation'][2]),
            np.cos(self.predicted_state['orientation'][2])
        )
    
    def _update_uncertainty(self):
        """
        更新不确定性
        """
        # 简化的不确定性更新
        # 假设不确定性随时间增加
        process_noise = np.diag([0.01, 0.01, 0.01, 0.005, 0.005, 0.005])
        self.uncertainty += process_noise
        
        # 限制不确定性的最大值
        max_uncertainty = 1.0
        self.uncertainty = np.clip(self.uncertainty, 0, max_uncertainty)
    
    def _fuse_perception_data(self):
        """
        融合感知数据
        """
        # 简化的感知数据融合
        # 假设最近的感知数据是最可靠的
        latest_perception = self.perception_buffer[-1]
        
        # 提取感知数据
        if 'pose' in latest_perception:
            perceived_pose = latest_perception['pose']
            # 简化的融合：加权平均
            weight = 0.7  # 感知数据的权重
            self.state_estimate['position'] = weight * perceived_pose[:3, 3] + (1 - weight) * self.predicted_state['position']
        
        if 'velocity' in latest_perception:
            perceived_velocity = latest_perception['velocity']
            weight = 0.7
            self.state_estimate['velocity'] = weight * perceived_velocity + (1 - weight) * self.predicted_state['velocity']
        
        # 清空感知缓冲区
        self.perception_buffer = []
        
        # 减少不确定性
        self.uncertainty *= 0.5
    
    def add_perception_data(self, perception_data):
        """
        添加感知数据
        
        Args:
            perception_data: 感知数据
        """
        self.perception_buffer.append(perception_data)
        # 限制缓冲区大小
        if len(self.perception_buffer) > 10:
            self.perception_buffer = self.perception_buffer[-10:]
    
    def get_state_estimate(self):
        """
        获取状态估计
        
        Returns:
            state_estimate: 状态估计
        """
        return self.state_estimate.copy()
    
    def get_predicted_state(self):
        """
        获取预测状态
        
        Returns:
            predicted_state: 预测状态
        """
        return self.predicted_state.copy()
    
    def get_uncertainty(self):
        """
        获取不确定性
        
        Returns:
            uncertainty: 不确定性
        """
        return self.uncertainty.copy()
    
    def compute_traversability(self, terrain_features):
        """
        计算地形可通行性
        
        Args:
            terrain_features: 地形特征
        
        Returns:
            traversability: 可通行性分数
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
        original_state = self.state_estimate.copy()
        original_predicted = self.predicted_state.copy()
        original_uncertainty = self.uncertainty.copy()
        original_buffer = self.perception_buffer.copy()
        
        try:
            # 执行预测
            for i in range(steps):
                state = self.step(wheel_commands, dt)
                predicted_states.append(state)
        finally:
            # 恢复原始状态
            self.state_estimate = original_state
            self.predicted_state = original_predicted
            self.uncertainty = original_uncertainty
            self.perception_buffer = original_buffer
        
        return predicted_states
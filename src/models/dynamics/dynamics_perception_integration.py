#!/usr/bin/env python3
"""
动力学与感知集成模块
将月球车动力学与感知系统集成
"""

import numpy as np

class ParameterEstimator:
    """
    参数估计器类
    基于感知数据在线辨识土壤参数
    实现递归最小二乘法(RLS)参数估计
    """
    
    def __init__(self):
        """
        初始化参数估计器
        """
        # 估计的土壤参数
        self.estimated_soil_params = {
            'kc': 2.9e4,      # cohesive modulus
            'kphi': 1.5e6,    # frictional modulus
            'n': 1.0,         # sinkage exponent
            'c': 1.1e3,       # cohesion
            'phi': 35         # internal friction angle
        }
        
        # 参数更新率
        self.learning_rate = 0.01
        
        # 历史滑移率
        self.slip_history = []
        
        # 历史加速度
        self.acceleration_history = []
        
        # 历史土壤参数
        self.param_history = []
        
        # 递归最小二乘法参数
        self.RLS_P = np.eye(2) * 100.0  # 协方差矩阵
        self.RLS_lambda = 0.99  # 遗忘因子
        
        print("参数估计器初始化完成")
    
    def update_parameters(self, measured_acceleration, predicted_acceleration, wheel_speeds, vehicle_velocity, soil_props):
        """
        使用感知数据更新土壤参数
        实现递归最小二乘法(RLS)参数估计
        
        Args:
            measured_acceleration: 测量的加速度
            predicted_acceleration: 预测的加速度
            wheel_speeds: 车轮角速度
            vehicle_velocity: 车辆速度
            soil_props: 当前土壤参数
        
        Returns:
            更新后的土壤参数
        """
        # 计算加速度误差
        acc_error = measured_acceleration - predicted_acceleration
        
        # 计算滑移率
        r = 0.25  # 车轮半径
        theoretical_velocity = r * np.mean(wheel_speeds)
        
        if abs(theoretical_velocity) < 1e-6:
            slip_ratio = 0.0
        else:
            slip_ratio = (theoretical_velocity - np.linalg.norm(vehicle_velocity)) / abs(theoretical_velocity)
            slip_ratio = max(-1.0, min(1.0, slip_ratio))
        
        # 存储历史数据
        self.slip_history.append(slip_ratio)
        self.acceleration_history.append(measured_acceleration)
        
        # 限制历史数据长度
        if len(self.slip_history) > 10:
            self.slip_history.pop(0)
        if len(self.acceleration_history) > 10:
            self.acceleration_history.pop(0)
        
        # 使用递归最小二乘法(RLS)更新参数
        error_magnitude = np.linalg.norm(acc_error)
        
        if error_magnitude > 0.01:
            # 构建观测向量 [slip_ratio, 1]
            phi = np.array([[abs(slip_ratio)], [1.0]])
            
            # RLS更新
            P_phi = self.RLS_P @ phi
            lambda_ = self.RLS_lambda
            denominator = lambda_ + phi.T @ P_phi
            gain = P_phi / denominator
            
            # 观测值（加速度误差的大小）
            y = abs(acc_error[0])
            
            # 计算预测值
            predicted_y = phi.T @ np.array([[self.estimated_soil_params['c']], [self.estimated_soil_params['phi']]])
            
            # 更新参数
            delta = gain * (y - predicted_y)
            self.estimated_soil_params['c'] += delta[0, 0] * 0.1
            self.estimated_soil_params['phi'] += delta[1, 0] * 0.01
            
            # 更新协方差矩阵
            self.RLS_P = (self.RLS_P - gain @ phi.T @ self.RLS_P) / lambda_
            
            # 限制参数在合理范围内
            self.estimated_soil_params['phi'] = max(20, min(45, self.estimated_soil_params['phi']))
            self.estimated_soil_params['c'] = max(0, min(2000, self.estimated_soil_params['c']))
        
        # 融合当前土壤参数和估计参数
        for key in self.estimated_soil_params:
            if key in soil_props:
                self.estimated_soil_params[key] = 0.7 * self.estimated_soil_params[key] + 0.3 * soil_props[key]
        
        # 存储参数历史
        self.param_history.append(self.estimated_soil_params.copy())
        if len(self.param_history) > 50:
            self.param_history.pop(0)
        
        return self.estimated_soil_params
    
    def get_estimated_params(self):
        """
        获取估计的土壤参数
        
        Returns:
            估计的土壤参数字典
        """
        return self.estimated_soil_params.copy()
    
    def detect_hazardous_terrain(self, slip_ratio):
        """
        检测危险地形
        
        Args:
            slip_ratio: 滑移率
        
        Returns:
            危险等级 (0-1)
        """
        if abs(slip_ratio) > 0.5:
            return 1.0  # 高危险
        elif abs(slip_ratio) > 0.3:
            return 0.7  # 中等危险
        elif abs(slip_ratio) > 0.1:
            return 0.3  # 低危险
        else:
            return 0.0  # 安全
    
    def get_param_history(self):
        """
        获取参数历史
        
        Returns:
            参数历史列表
        """
        return self.param_history.copy()
    
    def analyze_param_convergence(self):
        """
        分析参数收敛性
        
        Returns:
            收敛性分析结果
        """
        if len(self.param_history) < 5:
            return {
                'converged': False,
                'message': '历史数据不足'
            }
        
        # 计算最近10个参数的标准差
        recent_params = self.param_history[-10:]
        c_values = [p['c'] for p in recent_params]
        phi_values = [p['phi'] for p in recent_params]
        
        c_std = np.std(c_values)
        phi_std = np.std(phi_values)
        
        # 判断是否收敛
        c_converged = c_std < 50.0
        phi_converged = phi_std < 1.0
        
        return {
            'converged': c_converged and phi_converged,
            'c_std': c_std,
            'phi_std': phi_std,
            'c_mean': np.mean(c_values),
            'phi_mean': np.mean(phi_values),
            'message': f'c标准差: {c_std:.2f}, phi标准差: {phi_std:.2f}'
        }

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
        
        # 参数估计器
        self.parameter_estimator = ParameterEstimator()
        
        # 估计的土壤参数
        self.estimated_soil_params = self.parameter_estimator.estimated_soil_params.copy()
        
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
    
    def step(self, wheel_commands, dt, measured_acceleration=None, soil_props=None, dynamics_state=None):
        """
        执行集成模块一步
        实现数字孪生闭环
        
        Args:
            wheel_commands: 车轮控制命令
            dt: 时间步长
            measured_acceleration: 测量的加速度（可选）
            soil_props: 当前土壤参数（可选）
            dynamics_state: 动力学模型状态（可选）
        
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
        
        # 更新土壤参数（如果有测量数据）
        predicted_acceleration = np.array([0.0, 0.0, 0.0])
        
        if measured_acceleration is not None and soil_props is not None:
            # 从动力学状态获取预测加速度
            if dynamics_state and 'acceleration' in dynamics_state:
                predicted_acceleration = np.array([dynamics_state['acceleration'], 0.0, 0.0])
            
            # 更新土壤参数
            self.estimated_soil_params = self.parameter_estimator.update_parameters(
                measured_acceleration, 
                predicted_acceleration, 
                wheel_commands, 
                self.state_estimate['velocity'],
                soil_props
            )
        
        # 计算当前滑移率
        r = 0.25  # 车轮半径
        theoretical_velocity = r * np.mean(wheel_commands)
        
        if abs(theoretical_velocity) < 1e-6:
            current_slip_ratio = 0.0
        else:
            current_slip_ratio = (theoretical_velocity - np.linalg.norm(self.state_estimate['velocity'])) / abs(theoretical_velocity)
            current_slip_ratio = max(-1.0, min(1.0, current_slip_ratio))
        
        # 检测危险地形
        hazard_level = self.parameter_estimator.detect_hazardous_terrain(current_slip_ratio)
        
        # 分析参数收敛性
        convergence_analysis = self.parameter_estimator.analyze_param_convergence()
        
        # 构建集成状态
        integration_state = {
            'state_estimate': self.state_estimate.copy(),
            'predicted_state': self.predicted_state.copy(),
            'uncertainty': self.uncertainty.copy(),
            'perception_buffer_size': len(self.perception_buffer),
            'estimated_soil_params': self.estimated_soil_params.copy(),
            'slip_ratio': current_slip_ratio,
            'hazard_level': hazard_level,
            'convergence_analysis': convergence_analysis,
            'predicted_acceleration': predicted_acceleration,
            'measured_acceleration': measured_acceleration,
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
        left_speed = np.mean(wheel_commands[[0, 2, 4]])
        right_speed = np.mean(wheel_commands[[1, 3, 5]])
        
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
    
    def get_dynamics_features_for_perception(self):
        """
        获取动力学特征，用于感知系统
        
        Returns:
            动力学特征字典
        """
        return {
            'position': self.state_estimate['position'].tolist(),
            'velocity': self.state_estimate['velocity'].tolist(),
            'orientation': self.state_estimate['orientation'].tolist(),
            'angular_velocity': self.state_estimate['angular_velocity'].tolist(),
            'uncertainty': self.uncertainty.tolist(),
        }
    
    def get_environment_context(self):
        """
        获取环境上下文信息
        
        Returns:
            环境上下文字典
        """
        return {
            'predicted_state': {
                'position': self.predicted_state['position'].tolist(),
                'velocity': self.predicted_state['velocity'].tolist(),
            },
            'uncertainty': self.uncertainty.tolist(),
        }
    
    def get_camera_pose(self):
        """
        获取相机姿态信息
        
        Returns:
            相机姿态字典
        """
        return {
            'position': self.state_estimate['position'].tolist(),
            'orientation': self.state_estimate['orientation'].tolist(),
        }
    
    def get_terrain_perception_aids(self):
        """
        获取地形感知辅助信息
        
        Returns:
            地形感知辅助信息字典
        """
        return {
            'state_estimate': self.state_estimate.copy(),
            'predicted_state': self.predicted_state.copy(),
        }
    
    def get_navigation_features(self):
        """
        获取导航系统所需的特征
        
        Returns:
            导航特征字典
        """
        return {
            'position': self.state_estimate['position'].tolist(),
            'velocity': self.state_estimate['velocity'].tolist(),
            'orientation': self.state_estimate['orientation'].tolist(),
        }
    
    def calculate_motion_compensation(self):
        """
        计算运动补偿参数
        
        Returns:
            运动补偿参数字典
        """
        return {
            'velocity': self.state_estimate['velocity'].tolist(),
            'angular_velocity': self.state_estimate['angular_velocity'].tolist(),
        }
    
    def get_collision_risk(self, obstacles):
        """
        计算碰撞风险
        
        Args:
            obstacles: 障碍物列表
        
        Returns:
            带有碰撞风险的障碍物列表
        """
        # 简化的碰撞风险计算
        return obstacles
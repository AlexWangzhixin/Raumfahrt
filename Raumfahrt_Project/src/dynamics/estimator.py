#!/usr/bin/env python3
"""
参数估计器模块
用于在线估计月球车动力学和地面力学参数
"""

import numpy as np
import time

class ParameterEstimator:
    """
    参数估计器类
    用于在线更新摩擦系数等参数
    """
    
    def __init__(self, initial_params=None):
        """
        初始化参数估计器
        
        Args:
            initial_params: 初始参数
        """
        # 默认初始参数
        default_params = {
            'mu': 0.6,  # 摩擦系数
            'k_c': 0.1,  # cohesion modulus
            'k_phi': 10.0,  # frictional modulus
            'n': 1.1,  # 压力-沉陷指数
            'c': 0.5,  # 内聚力 (kPa)
            'phi': 30.0,  # 内摩擦角 (deg)
        }
        
        # 合并初始参数
        self.params = default_params.copy()
        if initial_params:
            self.params.update(initial_params)
        
        # 估计器参数
        self.learning_rate = 0.01  # 学习率
        self.history = []  # 历史数据
        self.window_size = 10  # 移动窗口大小，用于计算平均值
        self.confidence = 0.0  # 参数估计的置信度
        
        print("参数估计器初始化完成")
    
    def estimate_mu(self, measured_traction, predicted_traction, slip_ratio):
        """
        估计摩擦系数
        
        Args:
            measured_traction: 测量的牵引力 (N)
            predicted_traction: 预测的牵引力 (N)
            slip_ratio: 滑移率
        
        Returns:
            mu: 估计的摩擦系数
        """
        # 基于预测误差调整摩擦系数
        error = measured_traction - predicted_traction
        
        # 自适应学习率
        # 根据滑移率调整学习率，滑移率越大，学习率越小
        adaptive_learning_rate = self.learning_rate * (1 - min(abs(slip_ratio), 0.9))
        
        # 更新摩擦系数
        self.params['mu'] += adaptive_learning_rate * error * slip_ratio
        
        # 限制摩擦系数的范围
        self.params['mu'] = max(0.1, min(1.0, self.params['mu']))
        
        # 计算置信度
        self._update_confidence(error)
        
        # 记录历史
        self.history.append({
            'error': error,
            'mu': self.params['mu'],
            'slip_ratio': slip_ratio,
            'confidence': self.confidence,
            'timestamp': time.time(),
        })
        
        # 限制历史数据大小
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        return self.params['mu']
    
    def estimate_soil_parameters(self, sinkage, normal_load, wheel_radius=0.25, wheel_width=0.25):
        """
        估计土壤参数
        
        Args:
            sinkage: 沉陷量 (m)
            normal_load: 法向载荷 (N)
            wheel_radius: 车轮半径 (m)
            wheel_width: 车轮宽度 (m)
        
        Returns:
            soil_params: 估计的土壤参数
        """
        # 基于沉陷量和法向载荷估计k_c和k_phi
        b = wheel_width
        n = self.params['n']
        
        # 计算接触长度
        if sinkage < 2 * wheel_radius:
            contact_length = np.sqrt(2 * wheel_radius * sinkage - sinkage ** 2)
        else:
            contact_length = 2 * wheel_radius
        
        # 计算压力
        p = normal_load / (b * contact_length)
        
        # 估计k_c和k_phi
        k_c_phi = p / (sinkage ** n)
        
        # 假设k_c / b = k_phi
        k_phi = k_c_phi / 2
        k_c = k_phi * b
        
        # 更新参数
        self.params['k_c'] = k_c
        self.params['k_phi'] = k_phi
        
        return {
            'k_c': k_c,
            'k_phi': k_phi,
            'contact_length': contact_length,
            'pressure': p,
        }
    
    def estimate_all_parameters(self, sensor_data):
        """
        估计所有参数
        
        Args:
            sensor_data: 传感器数据字典，包含以下字段：
                - measured_traction: 测量的牵引力 (N)
                - predicted_traction: 预测的牵引力 (N)
                - slip_ratio: 滑移率
                - sinkage: 沉陷量 (m)
                - normal_load: 法向载荷 (N)
                - wheel_radius: 车轮半径 (m)
                - wheel_width: 车轮宽度 (m)
        
        Returns:
            params: 估计的所有参数
        """
        # 估计摩擦系数
        if all(key in sensor_data for key in ['measured_traction', 'predicted_traction', 'slip_ratio']):
            self.estimate_mu(
                sensor_data['measured_traction'],
                sensor_data['predicted_traction'],
                sensor_data['slip_ratio']
            )
        
        # 估计土壤参数
        if all(key in sensor_data for key in ['sinkage', 'normal_load']):
            wheel_radius = sensor_data.get('wheel_radius', 0.25)
            wheel_width = sensor_data.get('wheel_width', 0.25)
            self.estimate_soil_parameters(
                sensor_data['sinkage'],
                sensor_data['normal_load'],
                wheel_radius,
                wheel_width
            )
        
        return self.params.copy()
    
    def _update_confidence(self, error):
        """
        更新参数估计的置信度
        
        Args:
            error: 预测误差
        """
        # 基于预测误差的大小计算置信度
        # 误差越小，置信度越高
        error_magnitude = abs(error)
        max_error = 100.0  # 最大预期误差
        
        # 计算当前置信度
        current_confidence = max(0.0, 1.0 - error_magnitude / max_error)
        
        # 平滑置信度
        alpha = 0.1  # 平滑因子
        self.confidence = alpha * current_confidence + (1 - alpha) * self.confidence
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            params: 当前参数字典
        """
        return self.params.copy()
    
    def get_history(self):
        """
        获取历史数据
        
        Returns:
            history: 历史数据列表
        """
        return self.history.copy()
    
    def get_confidence(self):
        """
        获取参数估计的置信度
        
        Returns:
            confidence: 置信度 [0, 1]
        """
        return self.confidence
    
    def get_mu_convergence(self):
        """
        获取摩擦系数的收敛情况
        
        Returns:
            convergence: 收敛情况，包含以下字段：
                - converged: 是否已收敛
                - mean_mu: 最近窗口内的平均摩擦系数
                - std_mu: 最近窗口内的标准差
                - window_size: 窗口大小
        """
        if len(self.history) < self.window_size:
            return {
                'converged': False,
                'mean_mu': self.params['mu'],
                'std_mu': float('inf'),
                'window_size': len(self.history),
            }
        
        # 计算最近窗口内的摩擦系数平均值和标准差
        recent_mu = [h['mu'] for h in self.history[-self.window_size:]]
        mean_mu = np.mean(recent_mu)
        std_mu = np.std(recent_mu)
        
        # 判断是否收敛
        # 标准差小于0.01认为已收敛
        converged = std_mu < 0.01
        
        return {
            'converged': converged,
            'mean_mu': mean_mu,
            'std_mu': std_mu,
            'window_size': self.window_size,
        }
    
    def reset(self, initial_params=None):
        """
        重置参数估计器
        
        Args:
            initial_params: 新的初始参数
        """
        # 默认初始参数
        default_params = {
            'mu': 0.6,
            'k_c': 0.1,
            'k_phi': 10.0,
            'n': 1.1,
            'c': 0.5,
            'phi': 30.0,
        }
        
        # 合并初始参数
        self.params = default_params.copy()
        if initial_params:
            self.params.update(initial_params)
        
        # 重置历史数据和置信度
        self.history = []
        self.confidence = 0.0
        
        print("参数估计器重置完成")
        return self.params.copy()

class RecursiveLeastSquaresEstimator:
    """
    递归最小二乘参数估计器
    用于更准确地估计土壤和动力学参数
    """
    
    def __init__(self, initial_params=None, forgetting_factor=0.99):
        """
        初始化递归最小二乘估计器
        
        Args:
            initial_params: 初始参数
            forgetting_factor: 遗忘因子 (0-1)
        """
        # 默认初始参数
        default_params = {
            'k_c': 0.1,
            'k_phi': 10.0,
            'n': 1.1,
            'c': 0.5,
            'phi': 30.0,
            'mu': 0.6,
        }
        
        # 合并参数
        self.params = default_params.copy()
        if initial_params:
            self.params.update(initial_params)
        
        # 估计器参数
        self.forgetting_factor = forgetting_factor
        self.P = np.eye(6) * 100.0  # 初始协方差矩阵
        self.theta = np.array([
            self.params['k_c'],
            self.params['k_phi'],
            self.params['n'],
            self.params['c'],
            self.params['phi'],
            self.params['mu']
        ])
        
        # 历史数据
        self.history = []
        self.confidence = 0.0
        
        print("递归最小二乘估计器初始化完成")
    
    def estimate_parameters(self, measurements, inputs):
        """
        使用递归最小二乘法估计参数
        
        Args:
            measurements: 测量数据
            inputs: 输入数据
        
        Returns:
            theta: 估计的参数向量
        """
        # 递归最小二乘算法
        for y, u in zip(measurements, inputs):
            # 构建回归向量
            phi = self._build_regression_vector(u)
            
            # 计算增益矩阵
            P_phi = self.P @ phi
            lambda_inv = 1.0 / self.forgetting_factor
            denominator = lambda_inv + phi.T @ P_phi
            K = P_phi / denominator
            
            # 更新参数估计
            error = y - phi.T @ self.theta
            self.theta += K * error
            
            # 更新协方差矩阵
            self.P = (self.P - np.outer(K, phi.T @ self.P)) / self.forgetting_factor
            
            # 更新参数字典
            self.params['k_c'] = max(0.01, self.theta[0])
            self.params['k_phi'] = max(0.1, self.theta[1])
            self.params['n'] = max(0.5, min(2.0, self.theta[2]))
            self.params['c'] = max(0.01, self.theta[3])
            self.params['phi'] = max(10.0, min(45.0, self.theta[4]))
            self.params['mu'] = max(0.1, min(1.0, self.theta[5]))
            
            # 记录历史
            self.history.append({
                'theta': self.theta.copy(),
                'error': error,
                'timestamp': time.time(),
            })
        
        return self.theta
    
    def _build_regression_vector(self, inputs):
        """
        构建回归向量
        
        Args:
            inputs: 输入数据
        
        Returns:
            phi: 回归向量
        """
        # 这里需要根据具体的模型结构构建回归向量
        # 简化实现
        return np.array([
            inputs.get('sinkage', 0.0),
            inputs.get('normal_load', 0.0),
            inputs.get('slip_ratio', 0.0),
            inputs.get('velocity', 0.0),
            inputs.get('wheel_speed', 0.0),
            1.0
        ])
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            params: 当前参数字典
        """
        return self.params.copy()
    
    def get_covariance(self):
        """
        获取协方差矩阵
        
        Returns:
            P: 协方差矩阵
        """
        return self.P.copy()
    
    def get_history(self):
        """
        获取历史数据
        
        Returns:
            history: 历史数据列表
        """
        return self.history.copy()
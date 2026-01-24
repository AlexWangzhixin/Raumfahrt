#!/usr/bin/env python3
"""
地面力学模型测试用例
"""

import unittest
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.environment.terramechanics import Terramechanics, ParameterEstimator

class TestTerramechanics(unittest.TestCase):
    """
    地面力学模型测试类
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        # 初始化地面力学模型
        self.terramechanics = Terramechanics()
        # 初始化参数估计器
        self.parameter_estimator = ParameterEstimator()
        
    def test_calculate_sinkage(self):
        """
        测试沉陷量计算
        """
        # 测试不同法向载荷下的沉陷量
        normal_loads = [50, 100, 150, 200]
        sinkages = []
        
        for load in normal_loads:
            sinkage = self.terramechanics.calculate_sinkage(load)
            sinkages.append(sinkage)
            print(f"法向载荷 {load}N 时的沉陷量: {sinkage:.4f}m")
        
        # 验证沉陷量随载荷增加而增加
        self.assertTrue(all(sinkages[i] <= sinkages[i+1] for i in range(len(sinkages)-1)))
        
    def test_calculate_slip_ratio(self):
        """
        测试滑移率计算
        """
        # 测试不同车轮角速度和车辆速度下的滑移率
        wheel_angular_velocity = 10.0  # rad/s
        vehicle_velocities = [0.1, 0.5, 1.0, 1.5, 2.0]
        slip_ratios = []
        
        for velocity in vehicle_velocities:
            slip_ratio = self.terramechanics.calculate_slip_ratio(wheel_angular_velocity, velocity)
            slip_ratios.append(slip_ratio)
            print(f"车辆速度 {velocity}m/s 时的滑移率: {slip_ratio:.4f}")
        
        # 验证滑移率随车辆速度增加而减少
        self.assertTrue(all(slip_ratios[i] >= slip_ratios[i+1] for i in range(len(slip_ratios)-1)))
        
    def test_calculate_rolling_resistance(self):
        """
        测试滚动阻力计算
        """
        # 测试不同法向载荷和滑移率下的滚动阻力
        normal_load = 100  # N
        slip_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
        rolling_resistances = []
        
        for slip_ratio in slip_ratios:
            rolling_resistance = self.terramechanics.calculate_rolling_resistance(normal_load, slip_ratio)
            rolling_resistances.append(rolling_resistance)
            print(f"滑移率 {slip_ratio} 时的滚动阻力: {rolling_resistance:.4f}N")
        
        # 验证滚动阻力随滑移率增加而增加
        self.assertTrue(all(rolling_resistances[i] <= rolling_resistances[i+1] for i in range(len(rolling_resistances)-1)))
        
    def test_calculate_traction(self):
        """
        测试牵引力计算
        """
        # 测试不同切向载荷和滑移率下的牵引力
        tangential_load = 50  # N
        slip_ratios = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
        tractions = []
        
        for slip_ratio in slip_ratios:
            traction = self.terramechanics.calculate_traction(tangential_load, slip_ratio)
            tractions.append(traction)
            print(f"滑移率 {slip_ratio} 时的牵引力: {traction:.4f}N")
        
        # 验证负滑移（滑转）时牵引力为正
        for i, slip_ratio in enumerate(slip_ratios):
            if slip_ratio < 0:
                self.assertTrue(tractions[i] > 0)
        
        # 验证正滑移（滑移）时牵引力为负
        for i, slip_ratio in enumerate(slip_ratios):
            if slip_ratio > 0:
                self.assertTrue(tractions[i] < 0)
        
    def test_calculate_power_consumption(self):
        """
        测试功率消耗计算
        """
        # 测试不同牵引力、车辆速度和滚动阻力下的功率消耗
        tractions = [10, 20, 30, 40, 50]
        vehicle_velocity = 1.0  # m/s
        rolling_resistance = 10  # N
        power_consumptions = []
        
        for traction in tractions:
            power = self.terramechanics.calculate_power_consumption(traction, vehicle_velocity, rolling_resistance)
            power_consumptions.append(power)
            print(f"牵引力 {traction}N 时的功率消耗: {power:.4f}W")
        
        # 验证功率消耗随牵引力增加而增加
        self.assertTrue(all(power_consumptions[i] <= power_consumptions[i+1] for i in range(len(power_consumptions)-1)))
        
    def test_calculate_contact_length(self):
        """
        测试接触长度计算
        """
        # 测试不同沉陷量下的接触长度
        sinkages = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        contact_lengths = []
        
        for sinkage in sinkages:
            contact_length = self.terramechanics.calculate_contact_length(sinkage)
            contact_lengths.append(contact_length)
            print(f"沉陷量 {sinkage}m 时的接触长度: {contact_length:.4f}m")
        
        # 验证接触长度随沉陷量增加而增加
        self.assertTrue(all(contact_lengths[i] <= contact_lengths[i+1] for i in range(len(contact_lengths)-1)))
        
    def test_calculate_soil_stress(self):
        """
        测试土壤应力计算
        """
        # 测试不同法向载荷和沉陷量下的土壤应力
        normal_load = 100  # N
        sinkages = [0.01, 0.05, 0.1, 0.15, 0.2]
        soil_stresses = []
        
        for sinkage in sinkages:
            soil_stress = self.terramechanics.calculate_soil_stress(normal_load, sinkage)
            soil_stresses.append(soil_stress)
            print(f"沉陷量 {sinkage}m 时的土壤应力: {soil_stress:.4f}kPa")
        
        # 验证土壤应力随沉陷量增加而增加
        self.assertTrue(all(soil_stresses[i] <= soil_stresses[i+1] for i in range(len(soil_stresses)-1)))
        
    def test_calculate_traversability(self):
        """
        测试地形可通行性计算
        """
        # 测试不同地形特征下的可通行性
        terrain_features_list = [
            {'soil_type': 'loose_soil', 'roughness': 0.1, 'slope': 0.1, 'curvature': 0.05},
            {'soil_type': 'firm_soil', 'roughness': 0.05, 'slope': 0.05, 'curvature': 0.02},
            {'soil_type': 'rock', 'roughness': 0.2, 'slope': 0.15, 'curvature': 0.1},
        ]
        
        traversabilities = []
        for features in terrain_features_list:
            traversability = self.terramechanics.calculate_traversability(features)
            traversabilities.append(traversability)
            print(f"地形类型 {features['soil_type']} 时的可通行性: {traversability:.4f}")
        
        # 验证可通行性在[0, 1]范围内
        for traversability in traversabilities:
            self.assertTrue(0 <= traversability <= 1)
        
    def test_predict_energy_consumption(self):
        """
        测试路径能量消耗预测
        """
        # 生成测试路径
        path = [[0, 0], [10, 10], [20, 15], [30, 20], [40, 25]]
        terrain_map = None  # 使用默认地形地图
        
        # 预测能量消耗
        total_energy, energy_per_segment, energy_details = self.terramechanics.predict_energy_consumption(path, terrain_map)
        
        print(f"总能量消耗: {total_energy:.4f}J")
        print(f"每段路径的能量消耗: {energy_per_segment}")
        
        # 验证总能量消耗大于0
        self.assertTrue(total_energy > 0)
        # 验证每段路径的能量消耗大于0
        for energy in energy_per_segment:
            self.assertTrue(energy > 0)
        
    def test_calculate_path_cost(self):
        """
        测试路径成本计算
        """
        # 生成测试路径
        path = [[0, 0], [10, 10], [20, 15], [30, 20], [40, 25]]
        terrain_map = None  # 使用默认地形地图
        obstacles = None  # 无障碍物
        
        # 计算路径成本
        cost, cost_details = self.terramechanics.calculate_path_cost(path, terrain_map, obstacles)
        
        print(f"路径成本: {cost:.4f}")
        print(f"成本详细信息: {cost_details}")
        
        # 验证路径成本大于0
        self.assertTrue(cost > 0)
        
    def test_parameter_estimator(self):
        """
        测试参数估计器
        """
        # 测试摩擦系数估计
        measured_traction = 45.0  # N
        predicted_traction = 50.0  # N
        slip_ratio = -0.2
        
        # 估计摩擦系数
        mu_estimate = self.parameter_estimator.estimate_mu(measured_traction, predicted_traction, slip_ratio)
        print(f"估计的摩擦系数: {mu_estimate:.4f}")
        
        # 验证摩擦系数在[0.1, 1.0]范围内
        self.assertTrue(0.1 <= mu_estimate <= 1.0)
        
        # 测试土壤参数估计
        sinkage = 0.05  # m
        normal_load = 100  # N
        soil_params = self.parameter_estimator.estimate_soil_parameters(sinkage, normal_load)
        print(f"估计的土壤参数: {soil_params}")
        
        # 验证土壤参数大于0
        self.assertTrue(soil_params['k_c'] > 0)
        self.assertTrue(soil_params['k_phi'] > 0)

if __name__ == '__main__':
    unittest.main()
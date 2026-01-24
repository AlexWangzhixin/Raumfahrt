#!/usr/bin/env python3
"""
土壤参数数据库模块
包含月球土壤和其他行星土壤的物理参数
"""

import numpy as np

class SoilDatabase:
    """
    土壤参数数据库类
    提供不同类型土壤的物理参数
    """
    
    def __init__(self):
        """
        初始化土壤参数数据库
        """
        # 初始化土壤参数数据库
        self.soil_params = {
            # 月球土壤参数
            'lunar_regolith': {
                'name': '月球表土',
                'description': '月球表面的松散土壤，主要由玄武岩和斜长岩组成',
                'parameters': {
                    'k_c': 0.08,  # cohesion modulus (kPa/m^(n-1))
                    'k_phi': 9.5,  # frictional modulus (kPa/m^(n-1))
                    'n': 1.15,  # 压力-沉陷指数
                    'c': 0.4,  # 内聚力 (kPa)
                    'phi': 32.0,  # 内摩擦角 (deg)
                    'gamma': 1.62,  # 重力加速度 (m/s²)
                    'mu': 0.55,  # 车轮-土壤摩擦系数
                    'density': 1600,  # 密度 (kg/m³)
                    'porosity': 0.4,  # 孔隙率
                    'grain_size': 0.05,  # 颗粒大小 (mm)
                },
                'suitability': {
                    'traversal': 0.7,  # 通行性 [0-1]
                    'dust_formation': 0.9,  # 扬尘形成 [0-1]
                    'sinkage_risk': 0.6,  # 沉陷风险 [0-1]
                }
            },
            'lunar_mare_soil': {
                'name': '月球月海土壤',
                'description': '月球月海地区的土壤，主要由玄武岩组成',
                'parameters': {
                    'k_c': 0.06,  # cohesion modulus (kPa/m^(n-1))
                    'k_phi': 8.0,  # frictional modulus (kPa/m^(n-1))
                    'n': 1.2,  # 压力-沉陷指数
                    'c': 0.3,  # 内聚力 (kPa)
                    'phi': 30.0,  # 内摩擦角 (deg)
                    'gamma': 1.62,  # 重力加速度 (m/s²)
                    'mu': 0.5,  # 车轮-土壤摩擦系数
                    'density': 1500,  # 密度 (kg/m³)
                    'porosity': 0.45,  # 孔隙率
                    'grain_size': 0.04,  # 颗粒大小 (mm)
                },
                'suitability': {
                    'traversal': 0.65,  # 通行性 [0-1]
                    'dust_formation': 0.95,  # 扬尘形成 [0-1]
                    'sinkage_risk': 0.7,  # 沉陷风险 [0-1]
                }
            },
            'lunar_highland_soil': {
                'name': '月球高地土壤',
                'description': '月球高地地区的土壤，主要由斜长岩组成',
                'parameters': {
                    'k_c': 0.12,  # cohesion modulus (kPa/m^(n-1))
                    'k_phi': 12.0,  # frictional modulus (kPa/m^(n-1))
                    'n': 1.1,  # 压力-沉陷指数
                    'c': 0.6,  # 内聚力 (kPa)
                    'phi': 35.0,  # 内摩擦角 (deg)
                    'gamma': 1.62,  # 重力加速度 (m/s²)
                    'mu': 0.6,  # 车轮-土壤摩擦系数
                    'density': 1700,  # 密度 (kg/m³)
                    'porosity': 0.35,  # 孔隙率
                    'grain_size': 0.06,  # 颗粒大小 (mm)
                },
                'suitability': {
                    'traversal': 0.75,  # 通行性 [0-1]
                    'dust_formation': 0.85,  # 扬尘形成 [0-1]
                    'sinkage_risk': 0.5,  # 沉陷风险 [0-1]
                }
            },
            'lunar_rocky_soil': {
                'name': '月球岩石土壤',
                'description': '含有较多岩石碎片的月球土壤',
                'parameters': {
                    'k_c': 0.3,  # cohesion modulus (kPa/m^(n-1))
                    'k_phi': 25.0,  # frictional modulus (kPa/m^(n-1))
                    'n': 0.9,  # 压力-沉陷指数
                    'c': 1.5,  # 内聚力 (kPa)
                    'phi': 40.0,  # 内摩擦角 (deg)
                    'gamma': 1.62,  # 重力加速度 (m/s²)
                    'mu': 0.7,  # 车轮-土壤摩擦系数
                    'density': 1900,  # 密度 (kg/m³)
                    'porosity': 0.3,  # 孔隙率
                    'grain_size': 5.0,  # 颗粒大小 (mm)
                },
                'suitability': {
                    'traversal': 0.8,  # 通行性 [0-1]
                    'dust_formation': 0.6,  # 扬尘形成 [0-1]
                    'sinkage_risk': 0.3,  # 沉陷风险 [0-1]
                }
            },
            # 其他行星土壤参数
            'mars_soil': {
                'name': '火星土壤',
                'description': '火星表面的土壤，含有氧化铁',
                'parameters': {
                    'k_c': 0.15,  # cohesion modulus (kPa/m^(n-1))
                    'k_phi': 15.0,  # frictional modulus (kPa/m^(n-1))
                    'n': 1.05,  # 压力-沉陷指数
                    'c': 0.8,  # 内聚力 (kPa)
                    'phi': 33.0,  # 内摩擦角 (deg)
                    'gamma': 3.71,  # 重力加速度 (m/s²)
                    'mu': 0.58,  # 车轮-土壤摩擦系数
                    'density': 1700,  # 密度 (kg/m³)
                    'porosity': 0.4,  # 孔隙率
                    'grain_size': 0.07,  # 颗粒大小 (mm)
                },
                'suitability': {
                    'traversal': 0.7,  # 通行性 [0-1]
                    'dust_formation': 0.8,  # 扬尘形成 [0-1]
                    'sinkage_risk': 0.5,  # 沉陷风险 [0-1]
                }
            },
            'earth_sandy_soil': {
                'name': '地球沙土',
                'description': '地球表面的沙土',
                'parameters': {
                    'k_c': 0.05,  # cohesion modulus (kPa/m^(n-1))
                    'k_phi': 10.0,  # frictional modulus (kPa/m^(n-1))
                    'n': 1.0,  # 压力-沉陷指数
                    'c': 0.2,  # 内聚力 (kPa)
                    'phi': 30.0,  # 内摩擦角 (deg)
                    'gamma': 9.81,  # 重力加速度 (m/s²)
                    'mu': 0.4,  # 车轮-土壤摩擦系数
                    'density': 1600,  # 密度 (kg/m³)
                    'porosity': 0.4,  # 孔隙率
                    'grain_size': 0.2,  # 颗粒大小 (mm)
                },
                'suitability': {
                    'traversal': 0.6,  # 通行性 [0-1]
                    'dust_formation': 0.7,  # 扬尘形成 [0-1]
                    'sinkage_risk': 0.6,  # 沉陷风险 [0-1]
                }
            },
        }
        
        # 语义标签到土壤类型的映射
        self.semantic_mapping = {
            'loose_soil': 'lunar_mare_soil',
            'firm_soil': 'lunar_highland_soil',
            'rock': 'lunar_rocky_soil',
            'default': 'lunar_regolith',
        }
        
        print("土壤参数数据库初始化完成")
    
    def get_soil_parameters(self, soil_type):
        """
        获取指定土壤类型的参数
        
        Args:
            soil_type: 土壤类型
        
        Returns:
            params: 土壤参数字典
        """
        if soil_type in self.soil_params:
            return self.soil_params[soil_type]['parameters'].copy()
        else:
            # 返回默认参数
            return self.soil_params['lunar_regolith']['parameters'].copy()
    
    def get_soil_description(self, soil_type):
        """
        获取指定土壤类型的描述
        
        Args:
            soil_type: 土壤类型
        
        Returns:
            description: 土壤描述
        """
        if soil_type in self.soil_params:
            return self.soil_params[soil_type]['description']
        else:
            return "未知土壤类型"
    
    def get_soil_suitability(self, soil_type):
        """
        获取指定土壤类型的适合性评估
        
        Args:
            soil_type: 土壤类型
        
        Returns:
            suitability: 适合性评估字典
        """
        if soil_type in self.soil_params:
            return self.soil_params[soil_type]['suitability'].copy()
        else:
            return {
                'traversal': 0.5,
                'dust_formation': 0.5,
                'sinkage_risk': 0.5,
            }
    
    def get_soil_type_from_semantic(self, semantic_label):
        """
        根据语义标签获取土壤类型
        
        Args:
            semantic_label: 语义标签
        
        Returns:
            soil_type: 土壤类型
        """
        return self.semantic_mapping.get(semantic_label, 'default')
    
    def get_parameters_from_semantic(self, semantic_label):
        """
        根据语义标签获取土壤参数
        
        Args:
            semantic_label: 语义标签
        
        Returns:
            params: 土壤参数字典
        """
        soil_type = self.get_soil_type_from_semantic(semantic_label)
        return self.get_soil_parameters(soil_type)
    
    def list_available_soil_types(self):
        """
        列出所有可用的土壤类型
        
        Returns:
            soil_types: 土壤类型列表
        """
        return list(self.soil_params.keys())
    
    def get_soil_summary(self, soil_type):
        """
        获取土壤类型的摘要信息
        
        Args:
            soil_type: 土壤类型
        
        Returns:
            summary: 土壤摘要信息
        """
        if soil_type in self.soil_params:
            soil = self.soil_params[soil_type]
            summary = {
                'name': soil['name'],
                'description': soil['description'],
                'key_parameters': {
                    'cohesion': soil['parameters']['c'],
                    'friction_angle': soil['parameters']['phi'],
                    'pressure_sinkage_index': soil['parameters']['n'],
                    'density': soil['parameters']['density'],
                },
                'suitability': soil['suitability'],
            }
            return summary
        else:
            return {
                'name': '未知土壤',
                'description': '未知土壤类型',
                'key_parameters': {},
                'suitability': {},
            }
    
    def generate_random_soil_params(self, base_soil='lunar_regolith', variation=0.1):
        """
        生成带随机变化的土壤参数
        
        Args:
            base_soil: 基础土壤类型
            variation: 变化幅度 (0-1)
        
        Returns:
            params: 带随机变化的土壤参数字典
        """
        base_params = self.get_soil_parameters(base_soil)
        random_params = {}
        
        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                # 添加随机变化
                random_factor = 1.0 + np.random.uniform(-variation, variation)
                random_params[key] = value * random_factor
            else:
                random_params[key] = value
        
        return random_params
    
    def compare_soil_types(self, soil_type1, soil_type2):
        """
        比较两种土壤类型的参数
        
        Args:
            soil_type1: 第一种土壤类型
            soil_type2: 第二种土壤类型
        
        Returns:
            comparison: 比较结果
        """
        params1 = self.get_soil_parameters(soil_type1)
        params2 = self.get_soil_parameters(soil_type2)
        
        comparison = {
            'soil_type1': soil_type1,
            'soil_type2': soil_type2,
            'differences': {},
        }
        
        for key in params1:
            if key in params2:
                val1 = params1[key]
                val2 = params2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    percent_diff = (diff / val1) * 100 if val1 != 0 else 0
                    comparison['differences'][key] = {
                        'value1': val1,
                        'value2': val2,
                        'difference': diff,
                        'percent_difference': percent_diff,
                    }
        
        return comparison

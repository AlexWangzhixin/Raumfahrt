#!/usr/bin/env python3
"""
地面力学模型模块
基于Bekker理论的月球车轮-壤交互模型
"""

import numpy as np

class Terramechanics:
    """
    地面力学模型类
    实现基于Bekker理论的轮-壤交互模型
    """
    
    def __init__(self, parameters=None):
        """
        初始化地面力学模型
        
        Args:
            parameters: 地面力学参数
        """
        # 默认参数
        default_params = {
            # Bekker模型参数
            'k_c': 0.1,  #  cohesion模数 (kPa/m^(n-1))
            'k_phi': 10.0,  # 内摩擦模数 (kPa/m^(n-1))
            'n': 1.1,  # 压力-沉陷指数
            'c': 0.5,  # 内聚力 (kPa)
            'phi': 30.0,  # 内摩擦角 (deg)
            
            # Wong-Reece模型参数
            'b': 0.25,  # 车轮宽度 (m)
            'r': 0.25,  # 车轮半径 (m)
            'gamma': 1.62,  # 月球重力加速度 (m/s²)
            'mu': 0.6,  # 车轮-土壤摩擦系数
            
            # 滑移参数
            'slip_critical': 0.15,  # 临界滑移率
            'slip_max': 0.8,  # 最大滑移率
        }
        
        # 合并参数
        self.params = default_params.copy()
        if parameters:
            self.params.update(parameters)
        
        # 转换角度为弧度
        self.params['phi_rad'] = np.radians(self.params['phi'])
        
        print("地面力学模型初始化完成")
    
    def calculate_sinkage(self, normal_load):
        """
        计算沉陷量
        
        Args:
            normal_load: 法向载荷 (N)
        
        Returns:
            sinkage: 沉陷量 (m)
        """
        k_c = self.params['k_c']
        k_phi = self.params['k_phi']
        n = self.params['n']
        b = self.params['b']
        
        # Bekker公式: p = (k_c / b + k_phi) * z^n
        # 其中 p = normal_load / (b * l), l为接触长度
        # 简化计算，假设接触长度为车轮半径的一半
        l = self.params['r'] / 2
        p = normal_load / (b * l)
        
        # 计算沉陷量 z
        z = ((p) / (k_c / b + k_phi)) ** (1 / n)
        
        return z
    
    def calculate_rolling_resistance(self, normal_load, slip_ratio=0.0):
        """
        计算滚动阻力
        
        Args:
            normal_load: 法向载荷 (N)
            slip_ratio: 滑移率
        
        Returns:
            rolling_resistance: 滚动阻力 (N)
        """
        # 计算沉陷量
        z = self.calculate_sinkage(normal_load)
        
        # 简化的滚动阻力计算
        # 滚动阻力与沉陷量成正比
        rolling_resistance = normal_load * 0.1 * z
        
        # 滑移率增加滚动阻力
        rolling_resistance *= (1 + 5 * slip_ratio)
        
        return rolling_resistance
    
    def calculate_traction(self, tangential_load, slip_ratio):
        """
        计算牵引力
        
        Args:
            tangential_load: 切向载荷 (N)
            slip_ratio: 滑移率
        
        Returns:
            traction: 牵引力 (N)
        """
        # Wong-Reece模型简化版
        # 牵引力与滑移率的关系
        
        slip_critical = self.params['slip_critical']
        slip_max = self.params['slip_max']
        
        if slip_ratio < 0:
            # 负滑移（滑转）
            if abs(slip_ratio) < slip_critical:
                # 线性区域
                traction = tangential_load * (abs(slip_ratio) / slip_critical)
            else:
                # 非线性区域，逐渐饱和
                saturation = 0.95
                traction = tangential_load * (saturation - 0.1 * np.exp(-5 * (abs(slip_ratio) - slip_critical)))
        else:
            # 正滑移（滑移）
            traction = -tangential_load * slip_ratio
        
        return traction
    
    def calculate_slip_ratio(self, wheel_angular_velocity, vehicle_velocity):
        """
        计算滑移率
        
        Args:
            wheel_angular_velocity: 车轮角速度 (rad/s)
            vehicle_velocity: 车辆速度 (m/s)
        
        Returns:
            slip_ratio: 滑移率
        """
        r = self.params['r']
        
        # 计算理论车轮速度
        wheel_velocity = wheel_angular_velocity * r
        
        # 计算滑移率
        if vehicle_velocity > 0:
            slip_ratio = (wheel_velocity - vehicle_velocity) / vehicle_velocity
        else:
            slip_ratio = 0.0
        
        return slip_ratio
    
    def calculate_power_consumption(self, traction, vehicle_velocity, rolling_resistance):
        """
        计算功率消耗
        
        Args:
            traction: 牵引力 (N)
            vehicle_velocity: 车辆速度 (m/s)
            rolling_resistance: 滚动阻力 (N)
        
        Returns:
            power: 功率消耗 (W)
        """
        # 功率 = (牵引力 + 滚动阻力) * 速度
        power = (abs(traction) + rolling_resistance) * vehicle_velocity
        
        return power
    
    def calculate_contact_length(self, sinkage):
        """
        计算车轮与土壤的接触长度
        
        Args:
            sinkage: 沉陷量 (m)
        
        Returns:
            contact_length: 接触长度 (m)
        """
        r = self.params['r']
        
        # 接触长度计算
        # 基于几何关系: l = sqrt(2 * r * z - z^2)
        if sinkage < 2 * r:
            contact_length = np.sqrt(2 * r * sinkage - sinkage ** 2)
        else:
            # 沉陷量过大，接触长度等于车轮直径
            contact_length = 2 * r
        
        return contact_length
    
    def calculate_soil_stress(self, normal_load, sinkage):
        """
        计算土壤应力
        
        Args:
            normal_load: 法向载荷 (N)
            sinkage: 沉陷量 (m)
        
        Returns:
            soil_stress: 土壤应力 (kPa)
        """
        k_c = self.params['k_c']
        k_phi = self.params['k_phi']
        n = self.params['n']
        b = self.params['b']
        
        # Bekker公式: p = (k_c / b + k_phi) * z^n
        soil_stress = (k_c / b + k_phi) * (sinkage ** n)
        
        return soil_stress
    
    def calculate_traversability(self, terrain_features):
        """
        计算地形可通行性
        
        Args:
            terrain_features: 地形特征字典，包含粗糙度、坡度等
        
        Returns:
            traversability: 可通行性分数 [0, 1]
        """
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
        
        # 考虑土壤参数
        if 'soil_type' in terrain_features:
            soil_type = terrain_features['soil_type']
            soil_params = self.calculate_soil_parameters(soil_type)
            
            # 基于土壤参数调整可通行性
            if soil_type == 'loose_soil':
                traversability *= 0.7
            elif soil_type == 'rock':
                traversability *= 0.9
        
        return min(1.0, max(0.0, traversability))
    
    def predict_energy_consumption(self, path, terrain_map, rover_params=None):
        """
        预测路径的能量消耗

        Args:
            path: 路径点列表 [(x1, y1), (x2, y2), ...]
            terrain_map: 地形地图，包含每个位置的土壤参数
            rover_params: 月球车参数，包含质量、车轮半径等

        Returns:
            total_energy: 总能量消耗 (J)
            energy_per_segment: 每段路径的能量消耗 (J)
            energy_details: 能量消耗详细信息
        """
        if not path or len(path) < 2:
            return 0.0, [], {
                'length': [],
                'velocity': [],
                'power': [],
                'time': [],
                'energy': [],
                'terrain_type': [],
                'sinkage': [],
                'slip_ratio': [],
                'rolling_resistance': [],
                'traction': [],
            }

        total_energy = 0.0
        energy_per_segment = []
        energy_details = {
            'length': [],
            'velocity': [],
            'power': [],
            'time': [],
            'energy': [],
            'terrain_type': [],
            'sinkage': [],
            'slip_ratio': [],
            'rolling_resistance': [],
            'traction': [],
        }
        
        # 月球车参数
        default_rover_params = {
            'mass': 140.0,  # kg
            'wheel_radius': 0.25,  # m
            'wheel_width': 0.25,  # m
            'max_velocity': 0.3,  # 最大速度 m/s
        }
        
        if rover_params:
            default_rover_params.update(rover_params)
        
        mass = default_rover_params['mass']
        max_velocity = default_rover_params['max_velocity']
        
        for i in range(1, len(path)):
            try:
                # 计算路径段长度
                start = path[i-1]
                end = path[i]
                segment_length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                
                # 获取地形参数（简化为使用路径中点的地形参数）
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                
                # 获取地形类型
                terrain_type = self._get_terrain_type(mid_x, mid_y, terrain_map)
                soil_params = self.calculate_soil_parameters(terrain_type)
                
                # 更新地面力学模型参数
                self.update_parameters(soil_params)
                
                # 计算地形可通行性
                terrain_features = {'soil_type': terrain_type}
                traversability = self.calculate_traversability(terrain_features)
                
                # 根据可通行性调整速度
                velocity = max_velocity * traversability
                
                # 避免速度为零导致除零错误
                if velocity < 0.001:
                    velocity = 0.001
                
                # 计算法向载荷
                normal_load = mass * self.params['gamma']
                
                # 计算沉陷量
                sinkage = self.calculate_sinkage(normal_load)
                
                # 计算接触长度
                contact_length = self.calculate_contact_length(sinkage)
                
                # 计算滚动阻力
                rolling_resistance = self.calculate_rolling_resistance(normal_load)
                
                # 计算滑移率（简化为基于速度和地形的估计）
                slip_ratio = 0.05 * (1 - traversability)
                
                # 计算牵引力（假设需要克服滚动阻力和坡度阻力）
                # 简化计算，假设坡度阻力为0
                traction = rolling_resistance
                
                # 计算功率消耗
                power = self.calculate_power_consumption(traction, velocity, rolling_resistance)
                
                # 计算时间
                time = segment_length / velocity
                
                # 计算能量消耗
                energy = power * time
                total_energy += energy
                energy_per_segment.append(energy)
                
                # 记录详细信息
                energy_details['length'].append(segment_length)
                energy_details['velocity'].append(velocity)
                energy_details['power'].append(power)
                energy_details['time'].append(time)
                energy_details['energy'].append(energy)
                energy_details['terrain_type'].append(terrain_type)
                energy_details['sinkage'].append(sinkage)
                energy_details['slip_ratio'].append(slip_ratio)
                energy_details['rolling_resistance'].append(rolling_resistance)
                energy_details['traction'].append(traction)
            except Exception as e:
                print(f"计算能量消耗时出错: {e}")
                # 跳过错误的路径段
                continue
        
        return total_energy, energy_per_segment, energy_details
    
    def _get_terrain_type(self, x, y, terrain_map):
        """
        获取指定位置的地形类型
        
        Args:
            x: x坐标
            y: y坐标
            terrain_map: 地形地图
        
        Returns:
            terrain_type: 地形类型
        """
        # 简化实现
        # 在实际应用中，应该根据地形地图获取地形类型
        if terrain_map is None:
            # 随机生成地形类型，用于测试
            terrain_types = ['loose_soil', 'firm_soil', 'rock']
            return np.random.choice(terrain_types, p=[0.3, 0.6, 0.1])
        else:
            # 从地形地图获取地形类型
            # 这里需要根据实际的地形地图格式实现
            return 'firm_soil'  # 默认地形类型
    
    def calculate_path_cost(self, path, terrain_map, obstacles=None):
        """
        计算路径成本，考虑能耗和通过性
        
        Args:
            path: 路径点列表
            terrain_map: 地形地图
            obstacles: 障碍物列表
        
        Returns:
            cost: 路径成本
            cost_details: 成本详细信息
        """
        # 计算能量消耗
        total_energy, energy_per_segment, energy_details = self.predict_energy_consumption(path, terrain_map)
        
        # 计算路径长度
        path_length = 0.0
        for i in range(1, len(path)):
            start = path[i-1]
            end = path[i]
            path_length += np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        
        # 计算通过性
        total_traversability = 0.0
        traversability_values = []
        
        for i in range(len(path)):
            x, y = path[i]
            terrain_type = self._get_terrain_type(x, y, terrain_map)
            terrain_features = {'soil_type': terrain_type}
            traversability = self.calculate_traversability(terrain_features)
            total_traversability += traversability
            traversability_values.append(traversability)
        
        avg_traversability = total_traversability / len(path)
        
        # 计算障碍物成本
        obstacle_cost = 0.0
        if obstacles:
            for point in path:
                min_distance = float('inf')
                for obstacle in obstacles:
                    obs_pos = obstacle['position'][:2]
                    distance = np.sqrt((point[0] - obs_pos[0]) ** 2 + (point[1] - obs_pos[1]) ** 2)
                    min_distance = min(min_distance, distance)
                
                # 距离障碍物越近，成本越高
                if min_distance < 5.0:
                    obstacle_cost += (5.0 - min_distance) * 10
        
        # 计算总成本
        # 权重
        energy_weight = 0.5
        length_weight = 0.3
        traversability_weight = 0.1
        obstacle_weight = 0.1
        
        # 标准化各项成本
        max_energy = 10000.0  # 假设最大能量消耗为10000J
        normalized_energy = min(total_energy / max_energy, 1.0)
        
        max_length = 100.0  # 假设最大路径长度为100m
        normalized_length = min(path_length / max_length, 1.0)
        
        normalized_traversability = 1.0 - avg_traversability  # 可通行性越低，成本越高
        
        max_obstacle_cost = 1000.0  # 假设最大障碍物成本为1000
        normalized_obstacle = min(obstacle_cost / max_obstacle_cost, 1.0)
        
        # 计算加权成本
        cost = (
            energy_weight * normalized_energy +
            length_weight * normalized_length +
            traversability_weight * normalized_traversability +
            obstacle_weight * normalized_obstacle
        )
        
        cost_details = {
            'energy_cost': normalized_energy,
            'length_cost': normalized_length,
            'traversability_cost': normalized_traversability,
            'obstacle_cost': normalized_obstacle,
            'total_energy': total_energy,
            'path_length': path_length,
            'avg_traversability': avg_traversability,
            'obstacle_cost_raw': obstacle_cost,
        }
        
        return cost, cost_details
    
    def generate_energy_map(self, area_bounds, resolution, terrain_map):
        """
        生成能量消耗地图
        
        Args:
            area_bounds: 区域边界 [min_x, max_x, min_y, max_y]
            resolution: 分辨率 (m)
            terrain_map: 地形地图
        
        Returns:
            energy_map: 能量消耗地图
            traversability_map: 可通行性地图
        """
        min_x, max_x, min_y, max_y = area_bounds
        width = int((max_x - min_x) / resolution)
        height = int((max_y - min_y) / resolution)
        
        energy_map = np.zeros((height, width))
        traversability_map = np.zeros((height, width))
        
        # 月球车参数
        mass = 140.0  # kg
        velocity = 0.2  # m/s
        
        for i in range(height):
            for j in range(width):
                x = min_x + j * resolution
                y = min_y + i * resolution
                
                # 获取地形类型
                terrain_type = self._get_terrain_type(x, y, terrain_map)
                soil_params = self.calculate_soil_parameters(terrain_type)
                
                # 更新地面力学模型参数
                self.update_parameters(soil_params)
                
                # 计算地形可通行性
                terrain_features = {'soil_type': terrain_type}
                traversability = self.calculate_traversability(terrain_features)
                traversability_map[i, j] = traversability
                
                # 计算法向载荷
                normal_load = mass * self.params['gamma']
                
                # 计算沉陷量
                sinkage = self.calculate_sinkage(normal_load)
                
                # 计算滚动阻力
                rolling_resistance = self.calculate_rolling_resistance(normal_load)
                
                # 计算牵引力
                traction = rolling_resistance
                
                # 计算功率消耗
                power = self.calculate_power_consumption(traction, velocity, rolling_resistance)
                
                # 计算单位距离的能量消耗
                energy_per_meter = power / velocity
                energy_map[i, j] = energy_per_meter
        
        return energy_map, traversability_map
    
    def calculate_soil_parameters(self, semantic_label):
        """
        根据语义标签计算土壤参数
        
        Args:
            semantic_label: 语义标签
        
        Returns:
            soil_params: 土壤参数字典
        """
        # 语义标签到土壤参数的映射
        soil_param_map = {
            'loose_soil': {
                'k_c': 0.05,
                'k_phi': 8.0,
                'n': 1.2,
                'c': 0.3,
                'phi': 25.0,
                'mu': 0.5,
            },
            'firm_soil': {
                'k_c': 0.2,
                'k_phi': 15.0,
                'n': 1.0,
                'c': 0.8,
                'phi': 35.0,
                'mu': 0.7,
            },
            'rock': {
                'k_c': 1.0,
                'k_phi': 50.0,
                'n': 0.8,
                'c': 5.0,
                'phi': 45.0,
                'mu': 0.9,
            },
        }
        
        # 默认参数
        default_params = {
            'k_c': 0.1,
            'k_phi': 10.0,
            'n': 1.1,
            'c': 0.5,
            'phi': 30.0,
            'mu': 0.6,
        }
        
        # 获取对应的土壤参数
        soil_params = soil_param_map.get(semantic_label, default_params)
        
        return soil_params
    
    def update_parameters(self, new_params):
        """
        更新模型参数
        
        Args:
            new_params: 新的参数字典
        """
        self.params.update(new_params)
        
        # 转换角度为弧度
        if 'phi' in new_params:
            self.params['phi_rad'] = np.radians(new_params['phi'])
        
        print("地面力学模型参数更新完成")

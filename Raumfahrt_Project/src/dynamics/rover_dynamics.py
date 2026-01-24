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
    
    def __init__(self, rover_params=None, env_model=None):
        """
        初始化月球车动力学模型
        
        Args:
            rover_params: 月球车参数
            env_model: 环境模型实例
        """
        # 默认参数
        default_params = {
            'mass': 140.0,  # 质量 (kg)
            'wheel_radius': 0.25,  # 车轮半径 (m)
            'wheel_base': 1.5,  # 轴距 (m)
            'track_width': 1.0,  # 轮距 (m)
            'max_wheel_speed': 10.0,  # 最大车轮速度 (rad/s)
            'max_torque': 50.0,  # 最大扭矩 (N·m)
            'lunar_gravity': 1.62,  # 月球重力加速度 (m/s²)
            'inertia_tensor': np.diag([10.0, 10.0, 20.0]),  # 惯性张量
            'wheel_width': 0.15,  # 车轮宽度 (m)
        }
        
        # 合并参数
        self.params = default_params.copy()
        if rover_params:
            self.params.update(rover_params)
        
        # 环境模型引用
        self.env_model = env_model
        
        # 状态变量
        self.state = {
            'position': np.array([0.0, 0.0, 0.0]),  # 位置 (x, y, z)
            'velocity': np.array([0.0, 0.0, 0.0]),  # 速度 (x, y, z)
            'orientation': np.array([0.0, 0.0, 0.0]),  # 姿态 (roll, pitch, yaw)
            'angular_velocity': np.array([0.0, 0.0, 0.0]),  # 角速度
            'wheel_speeds': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 六个车轮的速度
        }
        
        # 能量消耗
        self.energy_consumed = 0.0
        
        # 接触状态
        self.contact_states = np.ones(6, dtype=bool)  # 六个车轮的接触状态
        
        # 滑移率
        self.slip_ratios = np.zeros(6)
        
        # 沉陷量
        self.sinkages = np.zeros(6)
        
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
        self.state['wheel_speeds'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.energy_consumed = 0.0
        self.contact_states = np.ones(6, dtype=bool)
        
        print(f"动力学模型重置完成，起始位置: {start_position}")
    
    def step(self, wheel_commands, dt):
        """
        执行动力学仿真一步
        
        Args:
            wheel_commands: 六个车轮的控制命令 (rad/s)
            dt: 时间步长 (s)
        
        Returns:
            state_info: 状态信息
        """
        # 限制车轮速度
        wheel_speeds = np.clip(wheel_commands, -self.params['max_wheel_speed'], self.params['max_wheel_speed'])
        self.state['wheel_speeds'] = wheel_speeds
        
        # 获取当前位置的土壤参数
        current_position = self.state['position']
        curr_x, curr_y = current_position[0], current_position[1]
        
        if self.env_model:
            # 使用新的get_soil_property方法获取土壤参数
            soil_props = self.env_model.get_soil_property(curr_x, curr_y)
        else:
            # 如果没有环境模型，使用默认参数（压实月壤）
            soil_props = {'kc': 2.9e4, 'kphi': 1.5e6, 'n': 1.0, 'c': 1.1e3, 'phi': 35}
        
        # 计算车身速度和方向
        yaw = self.state['orientation'][2]
        current_velocity = self.state['velocity'].copy()
        forward_velocity = np.linalg.norm(current_velocity[:2])
        
        # 计算每个车轮的负载（考虑车辆姿态的影响，简化为均匀分布）
        total_mass = self.params['mass']
        g = self.params['lunar_gravity']
        wheel_load = (total_mass * g) / 6.0  # 每个车轮的负载
        
        # 计算轮壤交互
        total_traction = 0.0
        total_resistance = 0.0
        total_torque = 0.0
        
        # 存储每个车轮的详细信息
        wheel_info = []
        
        for i in range(6):
            # 计算单轮的受力和滑移
            # 使用原有的calculate_wheel_soil_interaction方法
            traction, torque, sinkage, slip_ratio, rolling_resistance = self.calculate_wheel_soil_interaction(
                i, wheel_load, soil_props, wheel_speeds[i], forward_velocity
            )
            
            # 同时使用新的calculate_wheel_forces方法（用于对比和验证）
            wheel_radius = self.params['wheel_radius']
            wheel_linear_speed = wheel_speeds[i] * wheel_radius
            H, R, z, s = self.calculate_wheel_forces(
                wheel_linear_speed, forward_velocity, soil_props
            )
            
            # 累加到总牵引力和扭矩
            total_traction += traction
            total_resistance += rolling_resistance
            total_torque += abs(torque)
            
            # 更新滑移率和沉陷量
            self.slip_ratios[i] = slip_ratio
            self.sinkages[i] = sinkage
            
            # 存储车轮详细信息
            wheel_info.append({
                'wheel_idx': i,
                'slip_ratio': slip_ratio,
                'sinkage': sinkage,
                'traction': traction,
                'rolling_resistance': rolling_resistance,
                'torque': torque,
                'new_traction': H,
                'new_resistance': R,
                'new_sinkage': z,
                'new_slip': s
            })
        
        # 计算净牵引力
        net_traction = total_traction - total_resistance
        
        # 计算实际加速度（基于力平衡方程）
        # F = ma => a = F/m
        acceleration = net_traction / total_mass
        
        # 更新速度向量
        velocity = current_velocity.copy()
        velocity[0] += acceleration * np.cos(yaw) * dt
        velocity[1] += acceleration * np.sin(yaw) * dt
        velocity[2] = 0.0  # 假设在平面上运动
        
        # 计算角速度（基于左右轮速度差）
        left_speed = np.mean(wheel_speeds[[0, 2, 4]])  # 左侧车轮
        right_speed = np.mean(wheel_speeds[[1, 3, 5]])  # 右侧车轮
        r = self.params['wheel_radius']
        angular_velocity = (right_speed - left_speed) * r / self.params['track_width']
        
        # 更新状态
        self.state['velocity'] = velocity
        self.state['angular_velocity'][2] = angular_velocity
        
        # 更新位置
        self.state['position'] += velocity * dt
        
        # 更新姿态
        self.state['orientation'][2] += angular_velocity * dt
        # 限制偏航角在 [-π, π] 范围内
        self.state['orientation'][2] = np.arctan2(np.sin(self.state['orientation'][2]), np.cos(self.state['orientation'][2]))
        
        # 计算能量消耗（基于真实的扭矩和角速度）
        power = np.sum(np.abs(wheel_speeds) * np.abs(total_torque / 6.0))
        energy = power * dt
        self.energy_consumed += energy
        
        # 更新接触状态
        # 简化模型：假设所有车轮都与地面接触
        self.contact_states = np.ones(6, dtype=bool)
        
        # 构建状态信息
        state_info = {
            'position': self.state['position'].copy(),
            'velocity': self.state['velocity'].copy(),
            'orientation': self.state['orientation'].copy(),
            'angular_velocity': self.state['angular_velocity'].copy(),
            'wheel_speeds': self.state['wheel_speeds'].copy(),
            'energy_consumed': self.energy_consumed,
            'contact_states': self.contact_states.copy(),
            'slip_ratios': self.slip_ratios.copy(),
            'sinkages': self.sinkages.copy(),
            'soil_properties': soil_props,
            'total_traction': total_traction,
            'total_resistance': total_resistance,
            'net_traction': net_traction,
            'total_torque': total_torque,
            'acceleration': acceleration,
            'wheel_info': wheel_info
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
    
    def calculate_wheel_soil_interaction(self, wheel_idx, load_W, soil_props, wheel_speed, forward_velocity):
        """
        计算单轮的受力和滑移
        Args:
            wheel_idx: 车轮索引
            load_W: 轮上负载 (N)
            soil_props: 当前轮下的土壤参数 (kc, kphi, n, ...)
            wheel_speed: 车轮角速度 (rad/s)
            forward_velocity: 前进速度 (m/s)
        Returns:
            net_traction (净牵引力), torque (扭矩), sinkage (沉陷量), slip_ratio (滑移率), rolling_resistance (滚动阻力)
        """
        # 1. 沉陷量 z 计算 (Bekker公式)
        b = self.params['wheel_width'] # 车轮宽度
        kc, kphi, n = soil_props['kc'], soil_props['kphi'], soil_props['n']
        
        # 防止除零错误
        denominator = b * (kc/b + kphi)
        if abs(denominator) < 1e-12:
            sinkage_z = 0.0
        else:
            sinkage_z = (load_W / denominator) ** (1/n)
        
        # 2. 滚动阻力 R 计算（考虑沉陷的影响）
        if n + 1 > 0:
            rolling_resistance = (b * (kc/b + kphi) * (sinkage_z ** (n+1))) / (n+1)
        else:
            rolling_resistance = 0.0
        
        # 3. 计算滑移率
        r = self.params['wheel_radius']
        theoretical_velocity = r * wheel_speed
        
        if abs(theoretical_velocity) < 1e-6:
            slip_ratio = 0.0
        else:
            slip_ratio = (theoretical_velocity - forward_velocity) / abs(theoretical_velocity)
            slip_ratio = max(-1.0, min(1.0, slip_ratio))  # 限制在[-1, 1]范围内
        
        # 4. 驱动力 H 计算 (基于Janosi-Hanamoto公式)
        c, phi = soil_props['c'], soil_props['phi']
        
        # 剪切应力计算（考虑滑移率的影响）
        k = 0.6  # 剪切变形模量 (m)
        
        # 滑移率对剪切应力的影响
        slip_effect = 1.0 - np.exp(-abs(slip_ratio) * 10)  # 滑移率越大，剪切应力越大
        
        # 剪切应力计算
        tau = (c + load_W / (b * r) * np.tan(np.radians(phi))) * slip_effect * (1 - np.exp(-sinkage_z / k))
        
        # 驱动力
        drawbar_pull = tau * b * r
        
        # 5. 净牵引力 (驱动力减去滚动阻力)
        net_traction = drawbar_pull - rolling_resistance
        
        # 6. 扭矩计算
        torque = net_traction * r
        
        return net_traction, torque, sinkage_z, slip_ratio, rolling_resistance
    
    def calculate_wheel_forces(self, wheel_v, vehicle_v, soil_props):
        """
        计算单轮的受力 (基于 Bekker-Wong 理论)
        Args:
            wheel_v: 车轮线速度 (omega * r)
            vehicle_v: 车体实际速度
            soil_props: 土壤参数
        """
        # 1. 计算滑移率 (Slip Ratio)
        epsilon = 1e-5
        if abs(wheel_v) < epsilon:
            slip = 0.0
        else:
            slip = (wheel_v - vehicle_v) / wheel_v
            slip = np.clip(slip, -1.0, 1.0) # 限制范围
            
        # 2. 计算沉陷量 (Sinkage) z
        # 简化公式: W = (kc/b + kphi) * z^n * A
        # 这里为了简化，假设轮载荷固定为 W_wheel
        W_wheel = self.params['mass'] * 1.62 / 8.0 # 单轮载荷
        b = 0.15 # 轮宽
        kc, kphi, n = soil_props['kc'], soil_props['kphi'], soil_props['n']
        
        # z = (W / (b * (kc/b + kphi)))^(1/n)
        sinkage = (W_wheel / (b * (kc/b + kphi))) ** (1/n)
        
        # 3. 计算滚动阻力 (Rolling Resistance) R
        # R = (b * k * z^(n+1)) / (n+1)
        k_eq = kc/b + kphi
        resistance = (b * k_eq * (sinkage ** (n+1))) / (n+1)
        
        # 4. 计算牵引力 (Traction/Drawbar Pull) H
        # H = (Ac + W*tan(phi)) * (1 - exp(-slip/k_slip))
        # 简化版剪切模型
        c, phi = soil_props['c'], np.radians(soil_props['phi'])
        contact_area = b * np.sqrt(self.params['wheel_radius'] * sinkage) # 估算接触面积
        
        max_shear = contact_area * c + W_wheel * np.tan(phi)
        traction = max_shear * (1 - np.exp(-abs(slip) * 5.0)) * np.sign(slip)
        
        return traction, resistance, sinkage, slip
    
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
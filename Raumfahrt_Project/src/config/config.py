#!/usr/bin/env python3
"""
全局配置模块
定义物理常数、仿真步长等全局参数
"""

import numpy as np

# 物理常数
PHYSICAL_CONSTANTS = {
    'LUNAR_GRAVITY': 1.62,  # 月球重力加速度 (m/s²)
    'EARTH_GRAVITY': 9.81,   # 地球重力加速度 (m/s²)
    'AIR_DENSITY': 0.0,       # 月球大气密度 (kg/m³)
    'GRAVITATIONAL_CONSTANT': 6.67430e-11,  # 万有引力常数 (m³/kg/s²)
    'MOON_MASS': 7.342e22,    # 月球质量 (kg)
    'MOON_RADIUS': 1737.4e3,  # 月球半径 (m)
}

# 仿真参数
SIMULATION_PARAMS = {
    'TIME_STEP': 0.1,         # 仿真步长 (s)
    'SIMULATION_DURATION': 300,  # 最大仿真时间 (s)
    'MAX_STEPS': 3000,        # 最大仿真步数
    'COLLISION_THRESHOLD': 0.1,  # 碰撞检测阈值 (m)
}

# 环境参数
ENVIRONMENT_PARAMS = {
    'MAP_RESOLUTION': 0.1,    # 地图分辨率 (m/pixel)
    'MAX_ELEVATION': 10.0,     # 最大高程 (m)
    'MIN_ELEVATION': -5.0,     # 最小高程 (m)
    'SOIL_DENSITY': 1800,      # 月壤密度 (kg/m³)
    'SOIL_STIFFNESS': 10000,   # 月壤刚度 (Pa)
}

# 月球车参数
ROVER_PARAMS = {
    'MASS': 140.0,             # 质量 (kg)
    'WHEEL_RADIUS': 0.25,      # 车轮半径 (m)
    'WHEEL_WIDTH': 0.15,       # 车轮宽度 (m)
    'WHEEL_BASE': 1.5,         # 轴距 (m)
    'TRACK_WIDTH': 1.0,        # 轮距 (m)
    'MAX_WHEEL_SPEED': 10.0,   # 最大车轮速度 (rad/s)
    'MAX_TORQUE': 50.0,        # 最大扭矩 (N·m)
    'MAX_LINEAR_VELOCITY': 0.5,  # 最大线速度 (m/s)
    'MAX_ANGULAR_VELOCITY': 0.5,  # 最大角速度 (rad/s)
}

# 规划参数
PLANNING_PARAMS = {
    'ASTAR_HEURISTIC_WEIGHT': 1.0,  # A*启发函数权重
    'LOCAL_PLANNING_HORIZON': 10.0,  # 局部规划视野 (m)
    'OBSTACLE_PENALTY': 100.0,      # 障碍物惩罚
    'COLLISION_PENALTY': 500.0,     # 碰撞惩罚
    'GOAL_REWARD': 1000.0,          # 到达目标奖励
    'STEP_PENALTY': -0.1,           # 每步惩罚
    'SLIP_PENALTY': -1.0,           # 滑移惩罚
}

# 训练参数
TRAINING_PARAMS = {
    'BATCH_SIZE': 64,             # 批大小
    'GAMMA': 0.99,                # 折扣因子
    'EPS_START': 1.0,             # 初始探索率
    'EPS_END': 0.01,              # 最终探索率
    'EPS_DECAY': 0.995,           # 探索率衰减
    'TARGET_UPDATE': 1000,         # 目标网络更新频率
    'MEMORY_CAPACITY': 100000,     # 经验回放容量
    'LEARNING_RATE': 0.0001,       # 学习率
    'NUM_EPISODES': 10000,         # 训练 episodes
    'EVAL_FREQUENCY': 100,         # 评估频率
}

# 可视化参数
VISUALIZATION_PARAMS = {
    'FIGURE_SIZE': (12, 8),        # 图表大小
    'DPI': 100,                    # 图表DPI
    'COLOR_MAP': 'viridis',        # 颜色映射
    'TRAJECTORY_COLOR': '#1f77b4',  # 轨迹颜色
    'OBSTACLE_COLOR': '#d62728',   # 障碍物颜色
    'GOAL_COLOR': '#2ca02c',       # 目标颜色
    'START_COLOR': '#ff7f0e',      # 起点颜色
}

# 导出所有配置
__all__ = [
    'PHYSICAL_CONSTANTS',
    'SIMULATION_PARAMS',
    'ENVIRONMENT_PARAMS',
    'ROVER_PARAMS',
    'PLANNING_PARAMS',
    'TRAINING_PARAMS',
    'VISUALIZATION_PARAMS',
]

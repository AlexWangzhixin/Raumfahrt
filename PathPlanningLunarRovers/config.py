# -*- coding: utf-8 -*-
"""
配置文件 - 月球车路径规划系统参数配置
A*-D3QN-Opt: 基于数字孪生的月球车自主导航框架
"""

import numpy as np


class Config:
    """系统配置类"""

    # ==================== 环境参数 ====================
    # 环境尺寸 (米)
    ENV_WIDTH = 10.0
    ENV_HEIGHT = 10.0

    # 栅格地图分辨率 (米/像素)
    GRID_RESOLUTION = 0.1

    # 月球重力加速度 (m/s^2)
    LUNAR_GRAVITY = 1.62

    # 地形噪声参数 (高斯分布)
    TERRAIN_NOISE_MEAN = 0.1  # 米
    TERRAIN_NOISE_STD = 0.05  # 米

    # ==================== 月球车参数 ====================
    # 月球车质量 (kg)
    ROVER_MASS = 50.0

    # 转动惯量 (kg·m^2)
    ROVER_INERTIA = 15.0

    # 轴距 (m)
    ROVER_WHEELBASE = 0.4

    # 月球车尺寸 (m)
    ROVER_LENGTH = 0.6
    ROVER_WIDTH = 0.4

    # 月球车实际碰撞半径（用于碰撞检测）
    # 对角线半径 = sqrt(0.6^2+0.4^2)/2 ≈ 0.36m
    ROVER_COLLISION_RADIUS = 0.36

    # 月球车安全半径（用于A*路径规划膨胀）
    # 安全边距 = 对角线半径 + 0.1m裕度 = 0.46m
    # A*膨胀后总半径 = 0.3(障碍物) + 0.46 = 0.76m
    ROVER_SAFE_RADIUS = 0.46

    # 最大速度 (m/s)
    MAX_VELOCITY = 0.4

    # 最优节能速度 (m/s)
    OPTIMAL_VELOCITY = 0.3

    # ==================== 传感器参数 ====================
    # RGB-D相机参数
    CAMERA_FOCAL_LENGTH = 0.05  # 焦距 (m)
    CAMERA_BASELINE = 0.1  # 基线距离 (m)
    CAMERA_RESOLUTION = (640, 480)  # 分辨率

    # 深度图预处理尺寸
    DEPTH_IMAGE_SIZE = (80, 64)

    # 最大检测距离 (m)
    MAX_DETECTION_DISTANCE = 5.0

    # 传感器延迟 (ms)
    SENSOR_LATENCY = 80

    # ==================== D3QN网络参数 ====================
    # 动作空间大小 (9个离散动作)
    ACTION_SPACE_SIZE = 9

    # 动作定义: (速度, 转向角)
    # 低速动作用于精确避障，高速动作用于快速移动
    ACTIONS = [
        (0.2, np.pi / 6),    # 0: 低速左转30°
        (0.2, np.pi / 12),   # 1: 低速左转15°
        (0.2, 0.0),          # 2: 低速直行
        (0.2, -np.pi / 12),  # 3: 低速右转15°
        (0.2, -np.pi / 6),   # 4: 低速右转30°
        (0.4, np.pi / 12),   # 5: 高速左转15°
        (0.4, 0.0),          # 6: 高速直行
        (0.4, -np.pi / 12),  # 7: 高速右转15°
        (0.0, 0.0),          # 8: 紧急停止
    ]

    # 卷积层参数
    # Layer 1: 32个8x8卷积核, stride=4
    # Layer 2: 64个4x4卷积核, stride=2
    # Layer 3: 64个3x3卷积核, stride=1
    CONV_LAYERS = [
        {'filters': 32, 'kernel_size': 8, 'stride': 4},
        {'filters': 64, 'kernel_size': 4, 'stride': 2},
        {'filters': 64, 'kernel_size': 3, 'stride': 1},
    ]

    # 全连接层维度
    FC_HIDDEN_DIM = 512

    # ==================== 训练参数 ====================
    # 学习率（初始值，会随训练衰减）
    LEARNING_RATE = 0.0001

    # 学习率衰减参数
    LR_DECAY_RATE = 0.999     # 每回合衰减率（更慢衰减，约1000回合到最小值）
    LR_MIN = 0.00003          # 提高最小学习率，保持学习能力

    # 探索率参数 (ε-greedy)
    EPSILON_START = 1.0       # 初期100%探索
    EPSILON_MIN = 0.10        # 保持10%探索率，避免陷入局部最优
    EPSILON_DECAY = 0.998     # 更慢衰减，约1150回合到最小值

    # 折扣因子
    GAMMA = 0.99

    # 经验回放缓冲区大小
    REPLAY_BUFFER_SIZE = 200000

    # 小批量大小
    BATCH_SIZE = 64

    # 目标网络软更新参数
    USE_SOFT_UPDATE = True    # 使用软更新而非硬更新
    TAU = 0.005               # 软更新系数（提高以配合低频更新）
    TARGET_UPDATE_FREQ = 100  # 每100步更新一次目标网络（减少震荡）

    # 最大训练回合数
    MAX_EPISODES = 1000

    # 单回合最大步数
    MAX_STEPS_PER_EPISODE = 300

    # 随机种子
    RANDOM_SEED = 42

    # 梯度裁剪阈值
    GRAD_CLIP_NORM = 10.0

    # 训练开始前的预填充步数
    MIN_REPLAY_SIZE = 2000    # 预填充缓冲区（约15-20个episode后开始训练）

    # ==================== 优先经验回放参数 ====================
    # 优先级控制参数 α (0=随机采样, 1=完全按优先级)
    # 降低到0.3，进一步减少失败经验的过度采样
    PER_ALPHA = 0.3

    # 重要性采样参数 β
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0

    # TD误差小常数 (避免0优先级)
    PER_EPSILON = 1e-6

    # ==================== 奖励函数参数 ====================
    # 奖励设计原则：碰撞永远劣于成功，接近障碍物有明显惩罚
    #
    # 到达目标奖励（主要正向激励）
    REWARD_GOAL_REACHED = 500.0

    # 碰撞/出界惩罚
    # 设计依据：最大进度奖励≈450，碰撞惩罚需确保碰撞永远不如成功
    # 成功奖励≈650-700，碰撞后最多得 450-200=250 << 650
    REWARD_COLLISION = -200.0

    # 进度奖励系数（核心驱动力）
    # 起点到终点约11.3m，每步移动约0.04-0.08m
    # 进度奖励约 0.06 * 30 = 1.8/步（略微降低，平衡避障）
    REWARD_PROGRESS_WEIGHT = 30.0

    # 每步时间惩罚（小惩罚，不要淹没进度信号）
    REWARD_TIME_STEP = -0.1

    # 障碍物接近惩罚（持续性避障信号）
    # 最大惩罚（贴近障碍物）= -10/步，提供强避障梯度
    REWARD_OBSTACLE_PROXIMITY_WEIGHT = -10.0
    OBSTACLE_DANGER_DISTANCE = 1.2  # 扩大预警范围到1.2米

    # ==================== 障碍物参数 ====================
    # 静态障碍物数量 (Stage 2)
    NUM_STATIC_OBSTACLES = 4

    # 动态障碍物数量 (Stage 3)
    NUM_DYNAMIC_OBSTACLES = 4

    # 障碍物半径 (m)
    OBSTACLE_RADIUS = 0.3

    # 动态障碍物角速度 (rad/s)
    # 线速度 = 角速度 × 轨道半径 = 0.15 × 1.5 = 0.225 m/s（小于月球车0.4m/s）
    OBSTACLE_ANGULAR_VELOCITY = 0.15

    # 动态障碍物圆周运动半径 (m)
    OBSTACLE_ORBIT_RADIUS = 1.5

    # ==================== A*算法参数 ====================
    # 启发函数类型 ('manhattan', 'euclidean')
    ASTAR_HEURISTIC = 'manhattan'

    # 搜索方向 (4方向或8方向)
    ASTAR_DIRECTIONS = 8

    # ==================== 可视化参数 ====================
    # 是否实时显示
    ENABLE_VISUALIZATION = True

    # 可视化帧率
    VISUALIZATION_FPS = 30

    # 保存模型频率 (每N个episode)
    SAVE_MODEL_FREQ = 100

    # 日志目录
    LOG_DIR = './logs'

    # 模型保存目录
    MODEL_DIR = './models'

    # ==================== 深度图帧堆叠 ====================
    # 使用最近N帧深度图
    DEPTH_FRAME_STACK = 4


# 创建全局配置实例
config = Config()

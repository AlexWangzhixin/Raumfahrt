# -*- coding: utf-8 -*-
"""
月球环境仿真模块 (Digital Twin Environment)
实现月球表面数字孪生环境，包括地形、障碍物、月球车动力学模型
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from typing import Tuple, List, Optional, Dict
from collections import deque
from config import config

# ==================== 中文字体配置 ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Obstacle:
    """
    障碍物类
    支持静态和动态障碍物
    """

    def __init__(self, center: Tuple[float, float], radius: float,
                 is_dynamic: bool = False, orbit_center: Tuple[float, float] = None,
                 orbit_radius: float = None, angular_velocity: float = None,
                 initial_angle: float = 0.0):
        """
        初始化障碍物

        Args:
            center: 初始中心位置 (x, y)
            radius: 障碍物半径
            is_dynamic: 是否为动态障碍物
            orbit_center: 圆周运动中心（动态障碍物）
            orbit_radius: 圆周运动半径（动态障碍物）
            angular_velocity: 角速度 (rad/s)（动态障碍物）
            initial_angle: 初始角度 (rad)
        """
        self.radius = radius
        self.is_dynamic = is_dynamic

        if is_dynamic:
            self.orbit_center = orbit_center or center
            self.orbit_radius = orbit_radius or 1.5
            self.angular_velocity = angular_velocity or config.OBSTACLE_ANGULAR_VELOCITY
            self.current_angle = initial_angle
            # 根据初始角度计算初始位置
            self.x = self.orbit_center[0] + self.orbit_radius * np.cos(initial_angle)
            self.y = self.orbit_center[1] + self.orbit_radius * np.sin(initial_angle)
        else:
            self.x, self.y = center
            self.orbit_center = None
            self.orbit_radius = None
            self.angular_velocity = None
            self.current_angle = 0.0

    def update(self, dt: float):
        """
        更新障碍物位置（仅动态障碍物）

        Args:
            dt: 时间步长 (秒)
        """
        if self.is_dynamic:
            # 更新角度（顺时针旋转）
            self.current_angle += self.angular_velocity * dt
            # 保持角度在 [0, 2π) 范围内
            self.current_angle = self.current_angle % (2 * np.pi)
            # 更新位置
            self.x = self.orbit_center[0] + self.orbit_radius * np.cos(self.current_angle)
            self.y = self.orbit_center[1] + self.orbit_radius * np.sin(self.current_angle)

    @property
    def position(self) -> Tuple[float, float]:
        """返回当前位置"""
        return (self.x, self.y)


class LunarRover:
    """
    月球车类
    实现八轮月球车的运动学和动力学模型
    """

    def __init__(self, x: float = 0.5, y: float = 0.5, theta: float = 0.0):
        """
        初始化月球车

        Args:
            x: 初始x坐标 (m)
            y: 初始y坐标 (m)
            theta: 初始航向角 (rad)
        """
        # 位置状态
        self.x = x
        self.y = y
        self.theta = theta  # 航向角

        # 速度状态
        self.vx = 0.0  # x方向速度
        self.vy = 0.0  # y方向速度
        self.omega = 0.0  # 角速度

        # 物理参数
        self.mass = config.ROVER_MASS
        self.inertia = config.ROVER_INERTIA
        self.wheelbase = config.ROVER_WHEELBASE
        self.length = config.ROVER_LENGTH
        self.width = config.ROVER_WIDTH
        self.max_velocity = config.MAX_VELOCITY

        # 能量消耗记录
        self.total_energy = 0.0

    def reset(self, x: float = 0.5, y: float = 0.5, theta: float = 0.0):
        """
        重置月球车状态

        Args:
            x: 初始x坐标
            y: 初始y坐标
            theta: 初始航向角
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.total_energy = 0.0

    def step(self, velocity: float, steering_angle: float, dt: float = 0.1):
        """
        执行一个时间步的运动

        运动学模型:
        Vx = Σ(Vωi * cos(θ)) / n
        Vy = Σ(Vωi * sin(θ)) / n
        ω = (Vright - Vleft) / L

        Args:
            velocity: 目标速度 (m/s)
            steering_angle: 转向角 (rad)，正值为左转，负值为右转
            dt: 时间步长 (s)
        """
        # 限制速度范围
        velocity = np.clip(velocity, 0, self.max_velocity)

        # 更新角速度
        self.omega = velocity * np.tan(steering_angle) / self.wheelbase

        # 更新航向角
        self.theta += self.omega * dt
        # 归一化到 [-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # 更新速度分量
        self.vx = velocity * np.cos(self.theta)
        self.vy = velocity * np.sin(self.theta)

        # 更新位置
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 计算能量消耗（简化模型：与速度平方成正比）
        energy = 0.5 * self.mass * velocity ** 2 * dt
        self.total_energy += energy

    @property
    def position(self) -> Tuple[float, float]:
        """返回当前位置"""
        return (self.x, self.y)

    @property
    def state(self) -> Tuple[float, float, float]:
        """返回完整状态 (x, y, theta)"""
        return (self.x, self.y, self.theta)

    def get_corners(self) -> np.ndarray:
        """
        获取月球车四个角点的坐标（用于碰撞检测）

        Returns:
            4x2的角点坐标数组
        """
        half_length = self.length / 2
        half_width = self.width / 2

        # 相对于中心的角点（车体坐标系）
        corners_local = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        # 旋转矩阵
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # 转换到世界坐标系
        corners_world = np.dot(corners_local, rotation_matrix.T) + np.array([self.x, self.y])

        return corners_world


class LunarEnvironment:
    """
    月球环境类
    实现数字孪生仿真环境，包括地形、障碍物管理、碰撞检测等
    """

    def __init__(self, stage: int = 1, render: bool = True):
        """
        初始化月球环境

        Args:
            stage: 环境复杂度阶段
                   1 = 无障碍物 (Low complexity)
                   2 = 静态障碍物 (Medium complexity)
                   3 = 动态障碍物 (High complexity)
            render: 是否启用渲染
        """
        self.stage = stage
        self.render_enabled = render

        # 环境尺寸
        self.width = config.ENV_WIDTH
        self.height = config.ENV_HEIGHT
        self.resolution = config.GRID_RESOLUTION

        # 创建月球车
        self.rover = LunarRover()

        # 目标位置
        self.target_x = 9.0
        self.target_y = 9.0
        self.target_radius = 0.5  # 目标区域半径（增大便于到达）

        # 障碍物列表
        self.obstacles: List[Obstacle] = []
        self._setup_obstacles()

        # 深度图帧缓冲（存储最近4帧）
        self.depth_buffer = deque(maxlen=config.DEPTH_FRAME_STACK)

        # 时间步计数
        self.step_count = 0
        self.max_steps = config.MAX_STEPS_PER_EPISODE

        # 上一步到目标的距离（用于计算进度奖励）
        self.prev_distance = None

        # 碰撞计数
        self.collision_count = 0

        # 渲染相关
        self.fig = None
        self.ax = None

        # 轨迹记录（用于渲染）
        self.trajectory = []

        # A*路径（用于奖励计算）
        self.astar_path = None

        # 初始化环境
        self.reset()

    def _setup_obstacles(self):
        """根据阶段设置障碍物"""
        self.obstacles = []

        if self.stage == 1:
            # Stage 1: 无障碍物
            pass

        elif self.stage == 2:
            # Stage 2: 4个静态圆柱形障碍物
            static_positions = [
                (3.0, 3.0),
                (7.0, 3.0),
                (3.0, 7.0),
                (7.0, 7.0),
            ]
            for pos in static_positions:
                obstacle = Obstacle(
                    center=pos,
                    radius=config.OBSTACLE_RADIUS,
                    is_dynamic=False
                )
                self.obstacles.append(obstacle)

        elif self.stage == 3:
            # Stage 3: 4个动态障碍物（圆周运动）
            orbit_centers = [
                (3.0, 3.0),
                (7.0, 3.0),
                (3.0, 7.0),
                (7.0, 7.0),
            ]
            initial_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

            for i, (center, angle) in enumerate(zip(orbit_centers, initial_angles)):
                obstacle = Obstacle(
                    center=center,
                    radius=config.OBSTACLE_RADIUS,
                    is_dynamic=True,
                    orbit_center=center,
                    orbit_radius=config.OBSTACLE_ORBIT_RADIUS,
                    angular_velocity=config.OBSTACLE_ANGULAR_VELOCITY,
                    initial_angle=angle
                )
                self.obstacles.append(obstacle)

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns:
            初始状态
        """
        # 重置月球车位置（确保远离边界和障碍物）
        self.rover.reset(x=1.0, y=1.0, theta=0.0)  # 朝右，朝向目标大方向

        # 固定目标位置，让网络更容易学习稳定策略
        # 如需随机目标，可改为: np.random.uniform(8.0, 9.0)
        self.target_x = 9.0
        self.target_y = 9.0

        # 重置障碍物（动态障碍物回到初始位置）
        self._setup_obstacles()

        # 重置计数器
        self.step_count = 0
        self.collision_count = 0

        # 清空轨迹并记录起始位置
        self.trajectory = [(self.rover.x, self.rover.y)]

        # 计算初始距离
        self.prev_distance = self._get_distance_to_target()

        # 清空深度图缓冲并填充初始帧
        self.depth_buffer.clear()
        initial_depth = self._generate_depth_image()
        for _ in range(config.DEPTH_FRAME_STACK):
            self.depth_buffer.append(initial_depth)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作索引 (0-6)

        Returns:
            (next_state, reward, done, info)
        """
        # 获取动作对应的速度和转向角
        velocity, steering_angle = config.ACTIONS[action]

        # 更新动态障碍物位置
        dt = 0.2  # 时间步长 (增大以减少所需步数)
        for obstacle in self.obstacles:
            obstacle.update(dt)

        # 执行月球车运动
        self.rover.step(velocity, steering_angle, dt)

        # 记录轨迹
        self.trajectory.append((self.rover.x, self.rover.y))

        # 增加步数
        self.step_count += 1

        # 检查碰撞
        collision = self._check_collision()
        if collision:
            self.collision_count += 1

        # 检查是否到达目标
        reached_target = self._check_target_reached()

        # 检查是否超出边界
        out_of_bounds = self._check_out_of_bounds()

        # 计算奖励
        reward = self._calculate_reward(collision, reached_target, out_of_bounds)

        # 判断是否结束
        done = collision or reached_target or out_of_bounds or \
               self.step_count >= self.max_steps

        # 更新深度图缓冲
        depth_image = self._generate_depth_image()
        self.depth_buffer.append(depth_image)

        # 更新前一步距离
        self.prev_distance = self._get_distance_to_target()

        # 获取下一状态
        next_state = self._get_state()

        # 构建信息字典
        info = {
            'collision': collision,
            'reached_target': reached_target,
            'out_of_bounds': out_of_bounds,
            'step_count': self.step_count,
            'distance_to_target': self._get_distance_to_target(),
            'collision_count': self.collision_count,
            'energy_consumed': self.rover.total_energy,
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        获取当前状态

        状态向量: [深度图序列, 目标方向角, 目标距离]

        Returns:
            状态数组
        """
        # 堆叠深度图
        depth_stack = np.stack(list(self.depth_buffer), axis=-1)

        return depth_stack

    def get_additional_features(self) -> np.ndarray:
        """
        获取额外特征（包含目标、A*路径、360度障碍物信息）

        Returns:
            16维特征向量:
            [0] 目标方向角 (相对于当前航向, rad)
            [1] 目标距离 (m)
            [2] A*waypoint方向角 (相对于当前航向, rad)
            [3] A*waypoint距离 (m)
            [4] 路径偏离距离 (m)
            [5] A*路径有效标志 (0或1)
            [6] 最近障碍物方向角 (rad)
            [7] 最近障碍物距离 (m)
            [8-15] 8方向障碍物距离 (m, 归一化到0-1)
        """
        # 1. 计算目标方向角和距离
        theta_target = self._get_target_direction()
        distance = self._get_distance_to_target()

        # 2. 计算A* waypoint方向角和距离
        waypoint_angle = self._get_astar_waypoint_direction()
        waypoint_dist = self._get_astar_waypoint_distance()
        path_deviation = self._get_path_deviation_distance()

        # 判断A*路径是否有效
        astar_valid = 1.0 if (self.astar_path is not None and len(self.astar_path) > 0) else 0.0

        # 如果没有A*路径，使用目标方向作为默认值
        if waypoint_angle is None:
            waypoint_angle = theta_target
        if waypoint_dist is None:
            waypoint_dist = distance
        if path_deviation is None:
            path_deviation = 0.0

        # 3. 计算最近障碍物方向角和距离
        obs_angle, obs_dist = self._get_nearest_obstacle_info()

        # 4. 获取8方向障碍物距离（归一化到0-1）
        obstacle_distances = self.get_obstacle_distances()
        # 归一化：除以最大检测距离
        obstacle_distances_normalized = obstacle_distances / config.MAX_DETECTION_DISTANCE
        obstacle_distances_normalized = np.clip(obstacle_distances_normalized, 0, 1)

        # 组合所有特征
        base_features = np.array([theta_target, distance, waypoint_angle, waypoint_dist,
                                  path_deviation, astar_valid, obs_angle, obs_dist], dtype=np.float32)

        return np.concatenate([base_features, obstacle_distances_normalized.astype(np.float32)])

    def _get_nearest_obstacle_info(self) -> Tuple[float, float]:
        """
        获取最近障碍物的方向角和距离

        Returns:
            (方向角, 距离) - 方向角相对于当前航向，范围[-π, π]
        """
        min_dist = float('inf')
        nearest_obs = None

        for obs in self.obstacles:
            dx = obs.x - self.rover.x
            dy = obs.y - self.rover.y
            dist = np.sqrt(dx ** 2 + dy ** 2) - obs.radius
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs

        if nearest_obs is None:
            return 0.0, 5.0  # 无障碍物时返回默认值

        # 计算障碍物相对于月球车的方向角
        dx = nearest_obs.x - self.rover.x
        dy = nearest_obs.y - self.rover.y
        world_angle = np.arctan2(dy, dx)

        # 转换为相对于当前航向的角度
        relative_angle = world_angle - self.rover.theta
        # 归一化到 [-π, π]
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi

        return relative_angle, min_dist

    def _get_astar_waypoint_distance(self) -> float:
        """
        计算到A*路径下一个waypoint的距离

        Returns:
            距离（米），如果没有waypoint返回None
        """
        if self.astar_path is None or len(self.astar_path) == 0:
            return None

        # 找到最近的路径点
        min_dist = float('inf')
        nearest_idx = 0
        for i, wp in enumerate(self.astar_path):
            dx = wp[0] - self.rover.x
            dy = wp[1] - self.rover.y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 选择前方的路径点
        look_ahead = 3
        target_idx = min(nearest_idx + look_ahead, len(self.astar_path) - 1)
        waypoint = self.astar_path[target_idx]

        dx = waypoint[0] - self.rover.x
        dy = waypoint[1] - self.rover.y
        return np.sqrt(dx ** 2 + dy ** 2)

    def _get_path_deviation_distance(self) -> float:
        """
        计算当前位置到A*路径的最近距离（路径偏离度）

        这是A*信息的关键特征，帮助网络学习跟随规划路径

        Returns:
            到路径的最近距离（米），如果没有路径返回None
        """
        if self.astar_path is None or len(self.astar_path) == 0:
            return None

        min_dist = float('inf')
        for waypoint in self.astar_path:
            dx = self.rover.x - waypoint[0]
            dy = self.rover.y - waypoint[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _get_distance_to_target(self) -> float:
        """计算到目标的距离"""
        dx = self.target_x - self.rover.x
        dy = self.target_y - self.rover.y
        return np.sqrt(dx ** 2 + dy ** 2)

    def _get_target_direction(self) -> float:
        """
        计算目标方向角（相对于月球车当前航向）

        Returns:
            目标方向角 (rad)，范围 [-π, π]
        """
        # 目标相对于月球车的方向
        dx = self.target_x - self.rover.x
        dy = self.target_y - self.rover.y
        target_angle = np.arctan2(dy, dx)

        # 相对于当前航向的角度差
        angle_diff = target_angle - self.rover.theta

        # 归一化到 [-π, π]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        return angle_diff

    def _generate_depth_image(self) -> np.ndarray:
        """
        生成模拟深度图

        基于RGB-D相机参数模拟深度图生成:
        dp(x,y) = B * f / Z

        Returns:
            深度图数组 (80 x 64)
        """
        height, width = config.DEPTH_IMAGE_SIZE

        # 初始化深度图（最大检测距离）
        depth_image = np.ones((height, width), dtype=np.float32) * config.MAX_DETECTION_DISTANCE

        # 添加地形噪声
        noise = np.random.normal(
            config.TERRAIN_NOISE_MEAN,
            config.TERRAIN_NOISE_STD,
            (height, width)
        )
        depth_image += noise

        # 将障碍物投影到深度图
        for obstacle in self.obstacles:
            # 计算障碍物相对于月球车的位置
            dx = obstacle.x - self.rover.x
            dy = obstacle.y - self.rover.y

            # 转换到月球车坐标系
            cos_theta = np.cos(-self.rover.theta)
            sin_theta = np.sin(-self.rover.theta)
            local_x = dx * cos_theta - dy * sin_theta
            local_y = dx * sin_theta + dy * cos_theta

            # 只处理前方的障碍物
            if local_x <= 0:
                continue

            # 计算障碍物距离
            distance = np.sqrt(local_x ** 2 + local_y ** 2) - obstacle.radius

            if distance > 0 and distance < config.MAX_DETECTION_DISTANCE:
                # 计算障碍物在图像中的大致位置
                angle = np.arctan2(local_y, local_x)
                # 假设相机视场角为60度
                fov = np.pi / 3
                if abs(angle) < fov / 2:
                    # 计算像素位置
                    col = int(width / 2 + (angle / (fov / 2)) * (width / 2))
                    col = np.clip(col, 0, width - 1)

                    # 障碍物在图像中的大小（基于距离）
                    apparent_size = int(obstacle.radius / distance * height * 0.5)
                    apparent_size = max(1, min(apparent_size, height // 2))

                    # 更新深度图
                    row_start = max(0, height // 2 - apparent_size)
                    row_end = min(height, height // 2 + apparent_size)
                    col_start = max(0, col - apparent_size)
                    col_end = min(width, col + apparent_size)

                    depth_image[row_start:row_end, col_start:col_end] = \
                        np.minimum(depth_image[row_start:row_end, col_start:col_end], distance)

        # 归一化到 [0, 1]
        depth_image = depth_image / config.MAX_DETECTION_DISTANCE
        depth_image = np.clip(depth_image, 0, 1)

        return depth_image

    def _check_collision(self) -> bool:
        """
        检查是否发生碰撞

        使用月球车实际矩形形状进行精确碰撞检测

        Returns:
            True如果发生碰撞
        """
        # 月球车实际尺寸
        half_length = config.ROVER_LENGTH / 2  # 0.3m
        half_width = config.ROVER_WIDTH / 2    # 0.2m

        # 检查与边界的碰撞（使用矩形四个角点）
        corners = self._get_rover_corners()
        for cx, cy in corners:
            if cx < 0 or cx > self.width or cy < 0 or cy > self.height:
                return True

        # 检查与障碍物的碰撞（矩形-圆形碰撞检测）
        for obstacle in self.obstacles:
            if self._rect_circle_collision(obstacle.x, obstacle.y, obstacle.radius):
                return True

        return False

    def _get_rover_corners(self):
        """
        获取月球车矩形的四个角点坐标（考虑旋转）

        Returns:
            四个角点的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        half_length = config.ROVER_LENGTH / 2
        half_width = config.ROVER_WIDTH / 2
        theta = self.rover.theta

        # 旋转矩阵
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 四个角点（相对于中心的局部坐标）
        local_corners = [
            (half_length, half_width),    # 右前
            (half_length, -half_width),   # 右后
            (-half_length, -half_width),  # 左后
            (-half_length, half_width),   # 左前
        ]

        # 转换到世界坐标
        world_corners = []
        for lx, ly in local_corners:
            wx = self.rover.x + lx * cos_t - ly * sin_t
            wy = self.rover.y + lx * sin_t + ly * cos_t
            world_corners.append((wx, wy))

        return world_corners

    def _rect_circle_collision(self, circle_x: float, circle_y: float, circle_radius: float) -> bool:
        """
        检测旋转矩形与圆形是否碰撞

        算法：将圆心转换到矩形的局部坐标系，然后找到矩形上最近的点

        Args:
            circle_x: 圆心x坐标
            circle_y: 圆心y坐标
            circle_radius: 圆的半径

        Returns:
            True如果碰撞
        """
        half_length = config.ROVER_LENGTH / 2
        half_width = config.ROVER_WIDTH / 2
        theta = self.rover.theta

        # 将圆心转换到月球车局部坐标系
        dx = circle_x - self.rover.x
        dy = circle_y - self.rover.y

        # 反向旋转（将圆心转换到矩形对齐的坐标系）
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t

        # 找到矩形上距离圆心最近的点
        closest_x = np.clip(local_x, -half_length, half_length)
        closest_y = np.clip(local_y, -half_width, half_width)

        # 计算最近点到圆心的距离
        dist_x = local_x - closest_x
        dist_y = local_y - closest_y
        distance_squared = dist_x ** 2 + dist_y ** 2

        # 如果距离小于圆的半径，则碰撞
        return distance_squared < circle_radius ** 2

    def _check_target_reached(self) -> bool:
        """
        检查是否到达目标

        考虑月球车体积：月球车边缘碰到目标边缘即算到达

        Returns:
            True如果到达目标
        """
        distance = self._get_distance_to_target()
        # 使用统一的安全半径
        rover_radius = config.ROVER_SAFE_RADIUS
        # 月球车边缘到目标边缘的距离 < 0 即表示接触
        return distance < (self.target_radius + rover_radius)

    def _check_out_of_bounds(self) -> bool:
        """
        检查是否超出边界

        Returns:
            True如果超出边界
        """
        return (self.rover.x < 0 or self.rover.x > self.width or
                self.rover.y < 0 or self.rover.y > self.height)

    def _calculate_reward(self, collision: bool, reached_target: bool,
                          out_of_bounds: bool) -> float:
        """
        计算奖励函数（增强避障版）

        设计原则：
        - 主要依靠终止奖励和进度奖励
        - 添加障碍物接近惩罚，提供持续避障信号
        - 降低碰撞惩罚，减少经验分布偏斜

        奖励组件：
        - 到达目标：+500 + 时间奖励
        - 碰撞/出界：-50（降低以减少PER偏差）
        - 进度奖励：靠近目标得正奖励
        - 障碍物接近惩罚：距离越近惩罚越大（持续信号）
        - 每步小惩罚：-0.1（鼓励快速完成）
        """
        # 1. 终止条件奖励
        if reached_target:
            # 时间奖励：越快到达奖励越高
            time_bonus = max(0, (self.max_steps - self.step_count) / self.max_steps * 200)
            return config.REWARD_GOAL_REACHED + time_bonus

        if collision or out_of_bounds:
            return config.REWARD_COLLISION

        reward = 0.0

        # 2. 进度奖励（核心驱动力）
        current_distance = self._get_distance_to_target()
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance  # 正=靠近，负=远离
            # 使用较大的进度奖励系数，提供清晰的梯度信号
            reward += progress * config.REWARD_PROGRESS_WEIGHT

        # 3. 障碍物接近惩罚（持续性避障信号）
        min_obstacle_dist = self._get_min_obstacle_distance()
        if min_obstacle_dist < config.OBSTACLE_DANGER_DISTANCE:
            # 距离越近惩罚越大，使用线性惩罚
            # 当距离=0时惩罚最大，距离=DANGER_DISTANCE时惩罚=0
            proximity_penalty = (1.0 - min_obstacle_dist / config.OBSTACLE_DANGER_DISTANCE)
            reward += proximity_penalty * config.REWARD_OBSTACLE_PROXIMITY_WEIGHT

        # 4. 每步小惩罚（鼓励快速到达，但不要太大以免淹没进度信号）
        reward += config.REWARD_TIME_STEP

        return reward

    def _get_min_obstacle_distance(self) -> float:
        """获取到最近障碍物的距离"""
        min_dist = float('inf')
        for obs in self.obstacles:
            dx = obs.x - self.rover.x
            dy = obs.y - self.rover.y
            dist = np.sqrt(dx ** 2 + dy ** 2) - obs.radius
            min_dist = min(min_dist, dist)
        return min_dist

    def set_astar_path(self, path):
        """
        设置A*规划的路径（用于计算路径跟随奖励）

        Args:
            path: A*路径点列表 [(x, y), ...]
        """
        self.astar_path = path

    def _get_astar_waypoint_direction(self) -> float:
        """
        计算到A*路径下一个waypoint的方向角（相对于当前航向）

        Returns:
            方向角 (rad)，范围 [-π, π]，如果没有waypoint返回None
        """
        if self.astar_path is None or len(self.astar_path) == 0:
            return None

        # 找到最近的路径点
        min_dist = float('inf')
        nearest_idx = 0
        for i, wp in enumerate(self.astar_path):
            dx = wp[0] - self.rover.x
            dy = wp[1] - self.rover.y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 选择前方的路径点作为目标（跳过已经经过的）
        look_ahead = 3  # 前瞻点数
        target_idx = min(nearest_idx + look_ahead, len(self.astar_path) - 1)
        waypoint = self.astar_path[target_idx]

        # 计算方向角
        dx = waypoint[0] - self.rover.x
        dy = waypoint[1] - self.rover.y
        target_angle = np.arctan2(dy, dx)

        # 相对于当前航向的角度差
        angle_diff = target_angle - self.rover.theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        return angle_diff

    def _calculate_path_following_reward(self) -> float:
        """
        计算A*路径跟随奖励

        奖励靠近A*路径的行为，惩罚偏离路径的行为

        Returns:
            路径跟随奖励值
        """
        if self.astar_path is None or len(self.astar_path) == 0:
            return 0.0

        # 计算到A*路径的最小距离
        min_dist = float('inf')
        for waypoint in self.astar_path:
            dx = self.rover.x - waypoint[0]
            dy = self.rover.y - waypoint[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = min(min_dist, dist)

        # 路径跟随奖励：距离越近奖励越高
        # 在路径上（距离<0.5m）给正奖励，偏离路径给负奖励
        path_threshold = 0.5  # 认为"在路径上"的距离阈值
        if min_dist < path_threshold:
            # 在路径上，给正奖励
            path_reward = (1.0 - min_dist / path_threshold) * 2.0
        else:
            # 偏离路径，给负奖励（最大惩罚-2）
            path_reward = -min(min_dist - path_threshold, 2.0)

        return path_reward

    def get_obstacle_distances(self) -> np.ndarray:
        """
        获取8个方向的障碍物距离

        Returns:
            8个方向的最近障碍物距离数组
        """
        directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        distances = np.ones(8) * config.MAX_DETECTION_DISTANCE

        for i, angle in enumerate(directions):
            # 射线方向（世界坐标系）
            ray_angle = self.rover.theta + angle
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)

            # 检测该方向上的障碍物
            for obstacle in self.obstacles:
                # 计算射线与障碍物的最近距离
                ox = obstacle.x - self.rover.x
                oy = obstacle.y - self.rover.y

                # 投影到射线方向
                proj = ox * ray_dx + oy * ray_dy
                if proj > 0:  # 只考虑前方
                    # 垂直距离
                    perp = abs(ox * ray_dy - oy * ray_dx)
                    if perp < obstacle.radius:
                        # 计算实际距离
                        dist = proj - np.sqrt(obstacle.radius ** 2 - perp ** 2)
                        distances[i] = min(distances[i], max(0, dist))

            # 检测边界
            if ray_dx > 0:
                dist_to_wall = (self.width - self.rover.x) / ray_dx
                distances[i] = min(distances[i], dist_to_wall)
            elif ray_dx < 0:
                dist_to_wall = -self.rover.x / ray_dx
                distances[i] = min(distances[i], dist_to_wall)

            if ray_dy > 0:
                dist_to_wall = (self.height - self.rover.y) / ray_dy
                distances[i] = min(distances[i], dist_to_wall)
            elif ray_dy < 0:
                dist_to_wall = -self.rover.y / ray_dy
                distances[i] = min(distances[i], dist_to_wall)

        return distances

    def render(self, mode: str = 'human'):
        """
        渲染环境（简化版）

        Args:
            mode: 渲染模式
        """
        if not self.render_enabled:
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()

        # 设置坐标轴
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#3D3D3D')  # 月球灰色背景
        self.ax.grid(True, alpha=0.3, color='#606060')

        # 设置标签
        self.ax.set_xlabel('X (m)', fontsize=10)
        self.ax.set_ylabel('Y (m)', fontsize=10)
        stage_names = {1: '无障碍', 2: '静态障碍物', 3: '动态障碍物'}
        stage_name = stage_names.get(self.stage, f'Stage {self.stage}')
        self.ax.set_title(f'月球车数字孪生仿真 - {stage_name} (步数: {self.step_count})', fontsize=12)

        # 绘制A*规划路径（黄色虚线）
        if self.astar_path is not None and len(self.astar_path) > 1:
            path = np.array(self.astar_path)
            self.ax.plot(path[:, 0], path[:, 1], 'y--', linewidth=2, alpha=0.6, label='A*路径')

        # 绘制实际轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'c-', linewidth=1.5, alpha=0.7, label='实际轨迹')
            self.ax.plot(traj[0, 0], traj[0, 1], 'co', markersize=6)  # 起点

        # 绘制障碍物
        for obstacle in self.obstacles:
            if obstacle.is_dynamic:
                # 动态障碍物：红色
                circle = Circle((obstacle.x, obstacle.y), obstacle.radius,
                               facecolor='#CD5C5C', edgecolor='red', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)
                # 绘制轨道
                orbit = Circle(obstacle.orbit_center, obstacle.orbit_radius,
                              fill=False, linestyle='--', color='red', alpha=0.3)
                self.ax.add_patch(orbit)
            else:
                # 静态障碍物：深灰色
                circle = Circle((obstacle.x, obstacle.y), obstacle.radius,
                               facecolor='#2F2F2F', edgecolor='#1F1F1F', linewidth=2, alpha=0.9)
                self.ax.add_patch(circle)

        # 绘制目标
        target_circle = Circle((self.target_x, self.target_y), self.target_radius,
                               facecolor='#32CD32', edgecolor='green', linewidth=2, alpha=0.6)
        self.ax.add_patch(target_circle)
        self.ax.plot(self.target_x, self.target_y, 'g*', markersize=15)

        # 绘制月球车
        corners = self.rover.get_corners()
        rover_patch = plt.Polygon(corners, facecolor='#4169E1', edgecolor='blue',
                                   linewidth=2, alpha=0.9)
        self.ax.add_patch(rover_patch)

        # 方向箭头
        arrow_length = 0.5
        arrow_dx = arrow_length * np.cos(self.rover.theta)
        arrow_dy = arrow_length * np.sin(self.rover.theta)
        self.ax.arrow(self.rover.x, self.rover.y, arrow_dx, arrow_dy,
                     head_width=0.15, head_length=0.1, fc='yellow', ec='yellow')

        # 信息面板
        info_text = f'位置: ({self.rover.x:.2f}, {self.rover.y:.2f})\n'
        info_text += f'航向: {np.degrees(self.rover.theta):.1f}°\n'
        info_text += f'目标距离: {self._get_distance_to_target():.2f}m'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.pause(0.001)  # 1毫秒，更快刷新

    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# 测试代码
if __name__ == "__main__":
    import time

    print("=== 月球车仿真测试 ===")

    # 测试Stage 2（静态障碍物）
    env = LunarEnvironment(stage=2, render=True)
    env.reset()

    print(f"初始位置: ({env.rover.x:.2f}, {env.rover.y:.2f})")
    print(f"目标位置: ({env.target_x:.2f}, {env.target_y:.2f})")

    # 简单动作序列：持续向目标方向前进
    for step in range(60):
        action = 5 if step % 3 != 0 else 2  # 交替高速和低速直行
        next_state, reward, done, info = env.step(action)
        env.render()

        if done:
            print(f"结束: 碰撞={info['collision']}, 到达={info['reached_target']}")
            break

    print("测试完成，3秒后关闭...")
    time.sleep(3)
    env.close()

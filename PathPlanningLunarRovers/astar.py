# -*- coding: utf-8 -*-
"""
A*算法模块 - 静态全局路径规划
用于在静态环境中生成从起点到目标点的最优路径
"""

import heapq
import math
import numpy as np
from typing import List, Tuple, Optional, Set
from config import config


class Node:
    """
    A*算法节点类
    用于存储搜索过程中每个节点的信息
    """

    def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None):
        """
        初始化节点

        Args:
            position: 节点在栅格地图中的坐标 (x, y)
            parent: 父节点，用于回溯路径
        """
        self.position = position  # 节点位置
        self.parent = parent      # 父节点

        self.g = 0  # 从起点到当前节点的实际代价
        self.h = 0  # 从当前节点到目标的启发式估计代价
        self.f = 0  # 总代价 f = g + h

    def __lt__(self, other: 'Node') -> bool:
        """比较运算符，用于优先队列排序"""
        return self.f < other.f

    def __eq__(self, other: 'Node') -> bool:
        """相等运算符，用于判断节点是否相同"""
        return self.position == other.position

    def __hash__(self) -> int:
        """哈希函数，用于集合操作"""
        return hash(self.position)


class AStarPlanner:
    """
    A*路径规划器
    实现经典A*算法，用于在栅格地图中寻找最短路径
    """

    def __init__(self, grid_map: np.ndarray, resolution: float = 0.1):
        """
        初始化A*规划器

        Args:
            grid_map: 栅格地图，0表示可通行，1表示障碍物
            resolution: 地图分辨率 (米/格)
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.height, self.width = grid_map.shape

        # 搜索方向定义
        if config.ASTAR_DIRECTIONS == 8:
            # 8方向搜索 (包括对角线)
            self.directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            # 对应的移动代价 (对角线为√2)
            self.costs = [
                1.414, 1.0, 1.414,
                1.0,        1.0,
                1.414, 1.0, 1.414
            ]
        else:
            # 4方向搜索 (上下左右)
            self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.costs = [1.0, 1.0, 1.0, 1.0]

    def world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        将世界坐标转换为栅格坐标

        Args:
            world_pos: 世界坐标 (x, y) 单位：米

        Returns:
            栅格坐标 (row, col)
        """
        col = int(world_pos[0] / self.resolution)
        row = int(world_pos[1] / self.resolution)
        # 确保在地图范围内
        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width - 1))
        return (row, col)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        将栅格坐标转换为世界坐标

        Args:
            grid_pos: 栅格坐标 (row, col)

        Returns:
            世界坐标 (x, y) 单位：米
        """
        x = (grid_pos[1] + 0.5) * self.resolution
        y = (grid_pos[0] + 0.5) * self.resolution
        return (x, y)

    def heuristic(self, current: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        计算启发式函数值

        Args:
            current: 当前节点位置
            goal: 目标节点位置

        Returns:
            启发式估计值
        """
        if config.ASTAR_HEURISTIC == 'manhattan':
            # 曼哈顿距离: |x1-x2| + |y1-y2|
            return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
        else:
            # 欧几里得距离: sqrt((x1-x2)^2 + (y1-y2)^2)
            return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

    def is_valid(self, position: Tuple[int, int]) -> bool:
        """
        检查位置是否有效（在地图范围内且不是障碍物）

        Args:
            position: 待检查的位置

        Returns:
            True如果位置有效，否则False
        """
        row, col = position
        # 检查是否在地图范围内
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        # 检查是否是障碍物
        if self.grid_map[row, col] == 1:
            return False
        return True

    def get_neighbors(self, node: Node) -> List[Tuple[Tuple[int, int], float]]:
        """
        获取节点的所有有效邻居

        Args:
            node: 当前节点

        Returns:
            邻居列表，每个元素为 (位置, 移动代价)
        """
        neighbors = []
        for i, (dr, dc) in enumerate(self.directions):
            new_pos = (node.position[0] + dr, node.position[1] + dc)
            if self.is_valid(new_pos):
                neighbors.append((new_pos, self.costs[i]))
        return neighbors

    def _find_nearest_valid(self, grid_pos: Tuple[int, int], max_search: int = 10) -> Optional[Tuple[int, int]]:
        """
        找到最近的有效栅格位置

        Args:
            grid_pos: 原始栅格坐标
            max_search: 最大搜索半径

        Returns:
            最近的有效栅格坐标，如果找不到返回None
        """
        if self.is_valid(grid_pos):
            return grid_pos

        # 螺旋搜索最近的有效点
        for r in range(1, max_search + 1):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if abs(dr) == r or abs(dc) == r:  # 只检查外围
                        new_pos = (grid_pos[0] + dr, grid_pos[1] + dc)
                        if self.is_valid(new_pos):
                            return new_pos
        return None

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float], terrain_model=None) -> Optional[List[Tuple[float, float]]]:
        """
        使用A*算法寻找从起点到目标的最优路径
        考虑距离和能耗因素

        Args:
            start: 起点世界坐标 (x, y)
            goal: 目标世界坐标 (x, y)
            terrain_model: 地形模型，用于计算能耗

        Returns:
            路径点列表（世界坐标），如果找不到路径则返回None
        """
        # 转换为栅格坐标
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # 检查起点是否有效，如果无效则寻找最近的有效点
        if not self.is_valid(start_grid):
            start_grid = self._find_nearest_valid(start_grid)
            if start_grid is None:
                return None  # 静默失败，不打印警告

        # 检查终点是否有效，如果无效则寻找最近的有效点
        if not self.is_valid(goal_grid):
            goal_grid = self._find_nearest_valid(goal_grid)
            if goal_grid is None:
                return None

        # 创建起点和终点节点
        start_node = Node(start_grid)
        goal_node = Node(goal_grid)

        # 初始化开放列表（优先队列）和关闭列表（集合）
        open_list: List[Node] = []
        closed_set: Set[Tuple[int, int]] = set()

        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)

        # A*主循环
        while open_list:
            # 从开放列表中取出f值最小的节点
            current_node = heapq.heappop(open_list)

            # 如果当前节点已在关闭列表中，跳过
            if current_node.position in closed_set:
                continue

            # 将当前节点加入关闭列表
            closed_set.add(current_node.position)

            # 检查是否到达目标
            if current_node.position == goal_node.position:
                return self._reconstruct_path(current_node)

            # 遍历所有邻居
            for neighbor_pos, move_cost in self.get_neighbors(current_node):
                # 跳过已在关闭列表中的邻居
                if neighbor_pos in closed_set:
                    continue

                # 创建邻居节点
                neighbor_node = Node(neighbor_pos, current_node)

                # 计算g值（从起点到邻居的实际代价）
                # 基础代价：距离
                distance_cost = move_cost
                
                # 能耗代价（如果有地形模型）
                energy_cost = 0.0
                if terrain_model:
                    # 将栅格坐标转换为世界坐标
                    current_world = self.grid_to_world(current_node.position)
                    neighbor_world = self.grid_to_world(neighbor_pos)
                    
                    # 计算移动方向和距离
                    dx = neighbor_world[0] - current_world[0]
                    dy = neighbor_world[1] - current_world[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # 计算地形可通行性
                    terrain_features = {'soil_type': 'loose_soil'}  # 简化处理，实际应根据位置获取
                    traversability = terrain_model.calculate_traversability(terrain_features)
                    
                    # 计算能耗（基于第四章的能耗模型）
                    normal_load = 140.0 * 1.62 / 6  # 月球车质量140kg，月球重力1.62m/s²，6个车轮
                    sinkage = terrain_model.calculate_sinkage(normal_load)
                    slip_ratio = 0.05 * (1 - traversability)
                    rolling_resistance = terrain_model.calculate_rolling_resistance(normal_load, slip_ratio)
                    traction = rolling_resistance
                    velocity = 0.3 * traversability  # 基于可通行性的速度
                    
                    # 使用第四章的功率消耗公式
                    power = terrain_model.calculate_power_consumption(traction, velocity, rolling_resistance)
                    energy_cost = power * (distance / velocity) if velocity > 0 else 0
                
                # 总代价：距离 + 能耗权重 * 能耗
                total_cost = distance_cost + 0.1 * energy_cost
                neighbor_node.g = current_node.g + total_cost

                # 计算h值（从邻居到目标的启发式估计）
                neighbor_node.h = self.heuristic(neighbor_pos, goal_grid)

                # 计算f值
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # 将邻居加入开放列表
                heapq.heappush(open_list, neighbor_node)

        # 未找到路径
        print("警告: 未找到从起点到终点的有效路径")
        return None

    def _reconstruct_path(self, node: Node) -> List[Tuple[float, float]]:
        """
        从目标节点回溯重建路径

        Args:
            node: 目标节点

        Returns:
            路径点列表（世界坐标）
        """
        path = []
        current = node
        while current is not None:
            # 将栅格坐标转换为世界坐标
            world_pos = self.grid_to_world(current.position)
            path.append(world_pos)
            current = current.parent
        # 反转路径（从起点到终点）
        path.reverse()
        return path

    def smooth_path(self, path: List[Tuple[float, float]],
                    weight_data: float = 0.5,
                    weight_smooth: float = 0.1,
                    tolerance: float = 0.001) -> List[Tuple[float, float]]:
        """
        使用梯度下降法平滑路径

        Args:
            path: 原始路径点列表
            weight_data: 数据权重（保持原始路径）
            weight_smooth: 平滑权重（使路径更平滑）
            tolerance: 收敛容差

        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path

        # 复制路径
        new_path = [list(p) for p in path]
        change = tolerance + 1

        while change > tolerance:
            change = 0
            for i in range(1, len(new_path) - 1):
                for j in range(2):  # x和y坐标
                    old_val = new_path[i][j]
                    # 梯度下降更新
                    new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                    new_path[i][j] += weight_smooth * (
                        new_path[i - 1][j] + new_path[i + 1][j] - 2 * new_path[i][j]
                    )
                    change += abs(old_val - new_path[i][j])

        return [tuple(p) for p in new_path]

    def update_map(self, grid_map: np.ndarray):
        """
        更新栅格地图

        Args:
            grid_map: 新的栅格地图
        """
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape


def create_grid_map(width: float, height: float, resolution: float,
                    obstacles: List[Tuple[float, float, float]] = None,
                    robot_radius: float = 0.0) -> np.ndarray:
    """
    创建栅格地图（考虑机器人体积进行障碍物膨胀）

    Args:
        width: 地图宽度 (米)
        height: 地图高度 (米)
        resolution: 分辨率 (米/格)
        obstacles: 障碍物列表，每个元素为 (x, y, radius)
        robot_radius: 机器人安全半径，用于障碍物膨胀

    Returns:
        栅格地图数组
    """
    # 计算栅格尺寸
    grid_width = int(width / resolution)
    grid_height = int(height / resolution)

    # 初始化空地图（全部可通行）
    grid_map = np.zeros((grid_height, grid_width), dtype=np.int8)

    # 添加边界（考虑机器人半径膨胀）
    boundary_cells = int(robot_radius / resolution) + 1
    for i in range(boundary_cells):
        grid_map[i, :] = 1      # 上边界膨胀
        grid_map[-(i+1), :] = 1  # 下边界膨胀
        grid_map[:, i] = 1      # 左边界膨胀
        grid_map[:, -(i+1)] = 1  # 右边界膨胀

    # 添加障碍物（膨胀后）
    if obstacles:
        for ox, oy, obs_radius in obstacles:
            # 膨胀后的障碍物半径 = 原始半径 + 机器人安全半径
            inflated_radius = obs_radius + robot_radius

            # 将障碍物中心转换为栅格坐标
            center_col = int(ox / resolution)
            center_row = int(oy / resolution)
            # 膨胀后半径对应的栅格数
            radius_cells = int(inflated_radius / resolution) + 1

            # 标记障碍物区域
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    row = center_row + dr
                    col = center_col + dc
                    # 检查是否在地图范围内
                    if 0 <= row < grid_height and 0 <= col < grid_width:
                        # 检查是否在膨胀后的圆形障碍物内
                        dist = np.sqrt(dr ** 2 + dc ** 2) * resolution
                        if dist <= inflated_radius:
                            grid_map[row, col] = 1

    return grid_map


# 测试代码
if __name__ == "__main__":
    # 创建测试地图
    obstacles = [
        (3.0, 3.0, 0.5),
        (5.0, 5.0, 0.5),
        (7.0, 3.0, 0.5),
        (3.0, 7.0, 0.5),
    ]
    grid_map = create_grid_map(10.0, 10.0, 0.1, obstacles)

    # 创建A*规划器
    planner = AStarPlanner(grid_map, resolution=0.1)

    # 规划路径
    start = (0.5, 0.5)
    goal = (9.5, 9.5)
    path = planner.find_path(start, goal)

    if path:
        print(f"找到路径，共 {len(path)} 个路径点")
        # 平滑路径
        smooth_path = planner.smooth_path(path)
        print(f"平滑后路径，共 {len(smooth_path)} 个路径点")
    else:
        print("未找到路径")

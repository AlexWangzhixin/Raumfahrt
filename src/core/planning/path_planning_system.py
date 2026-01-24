#!/usr/bin/env python3
"""
路径规划系统模块
基于嫦娥6号数据的月球车路径规划
"""

import numpy as np

# 导入地面力学模型
import sys
sys.path.append('..')
sys.path.append('../..')
from src.models.environment.terramechanics import Terramechanics

class PathPlanningSystem:
    """
    路径规划系统类
    """
    
    def __init__(self, planner_params=None):
        """
        初始化路径规划系统
        
        Args:
            planner_params: 规划器参数
        """
        # 默认参数
        default_params = {
            'grid_resolution': 0.5,  # 网格分辨率 (m)
            'max_planning_time': 10.0,  # 最大规划时间 (s)
            'robot_radius': 0.5,  # 机器人半径 (m)
            'safety_margin': 0.5,  # 安全 margin (m)
            'max_iterations': 1000,  # 最大迭代次数
        }
        
        # 合并参数
        self.params = default_params.copy()
        if planner_params:
            self.params.update(planner_params)
        
        # 初始化地面力学模型
        self.terramechanics = Terramechanics()
        
        # 规划结果
        self.plan_result = {
            'status': 'idle',
            'path': [],
            'path_length': 0.0,
            'waypoints': [],
            'estimated_time': 0.0,
            'cost': 0.0,
            'message': '',
        }
        
        print("路径规划系统初始化完成")
    
    def plan_path(self, start_position, goal_position, obstacles=None, terrain_map=None):
        """
        规划路径
        
        Args:
            start_position: 起始位置 [x, y, z]
            goal_position: 目标位置 [x, y, z]
            obstacles: 障碍物列表
            terrain_map: 地形地图
        
        Returns:
            plan_result: 规划结果
        """
        print(f"开始路径规划: 起点={start_position}, 终点={goal_position}")
        
        # 重置规划结果
        self.plan_result = {
            'status': 'planning',
            'path': [],
            'path_length': 0.0,
            'waypoints': [],
            'estimated_time': 0.0,
            'cost': 0.0,
            'message': '',
        }
        
        try:
            # 简化的路径规划算法
            # 使用A*算法的简化版本
            path = self._simple_astar(start_position, goal_position, obstacles)
            
            if path:
                # 计算路径长度
                path_length = self._calculate_path_length(path)
                
                # 生成航点
                waypoints = self._generate_waypoints(path)
                
                # 计算估计时间
                estimated_time = path_length / 0.1  # 假设速度为0.1 m/s
                
                # 计算路径成本
                cost = self._calculate_path_cost(path, obstacles, terrain_map)
                
                # 更新规划结果
                self.plan_result = {
                    'status': 'success',
                    'path': path,
                    'path_length': path_length,
                    'waypoints': waypoints,
                    'estimated_time': estimated_time,
                    'cost': cost,
                    'message': '路径规划成功',
                }
                
                print(f"路径规划成功: 路径长度={path_length:.2f}m, 航点数量={len(waypoints)}")
            else:
                self.plan_result = {
                    'status': 'failed',
                    'path': [],
                    'path_length': 0.0,
                    'waypoints': [],
                    'estimated_time': 0.0,
                    'cost': 0.0,
                    'message': '无法找到可行路径',
                }
                
                print("路径规划失败: 无法找到可行路径")
                
        except Exception as e:
            self.plan_result = {
                'status': 'error',
                'path': [],
                'path_length': 0.0,
                'waypoints': [],
                'estimated_time': 0.0,
                'cost': 0.0,
                'message': f'规划过程中发生错误: {e}',
            }
            
            print(f"路径规划错误: {e}")
        
        return self.plan_result
    
    def _simple_astar(self, start, goal, obstacles=None):
        """
        简化的A*算法
        
        Args:
            start: 起始位置
            goal: 目标位置
            obstacles: 障碍物列表
        
        Returns:
            path: 规划的路径
        """
        # 简化版本：直接生成直线路径，避开障碍物
        path = [start[:2], goal[:2]]
        
        # 检查路径是否与障碍物碰撞
        if obstacles:
            # 简化的碰撞检测
            collision_free = True
            for obstacle in obstacles:
                obs_pos = obstacle['position'][:2]
                obs_size = obstacle['size'][:2]
                
                # 检查路径是否与障碍物碰撞
                if self._line_obstacle_collision(start[:2], goal[:2], obs_pos, obs_size):
                    collision_free = False
                    break
            
            if not collision_free:
                # 如果碰撞，生成绕过障碍物的路径
                path = self._generate_detour_path(start[:2], goal[:2], obstacles)
        
        return path
    
    def _line_obstacle_collision(self, start, end, obstacle_pos, obstacle_size):
        """
        检查线段是否与障碍物碰撞
        
        Args:
            start: 线段起点
            end: 线段终点
            obstacle_pos: 障碍物位置
            obstacle_size: 障碍物大小
        
        Returns:
            collision: 是否碰撞
        """
        # 简化的碰撞检测：检查线段是否与障碍物的矩形边界框相交
        # 计算线段的参数方程
        def line_segment_rect_intersection(p1, p2, rect_center, rect_size):
            """检查线段是否与矩形相交"""
            x_min = rect_center[0] - rect_size[0]/2 - self.params['safety_margin']
            x_max = rect_center[0] + rect_size[0]/2 + self.params['safety_margin']
            y_min = rect_center[1] - rect_size[1]/2 - self.params['safety_margin']
            y_max = rect_center[1] + rect_size[1]/2 + self.params['safety_margin']
            
            # 线段的边界框
            x1, y1 = p1
            x2, y2 = p2
            
            # 快速排斥实验
            if (max(x1, x2) < x_min or min(x1, x2) > x_max or
                max(y1, y2) < y_min or min(y1, y2) > y_max):
                return False
            
            # 跨立实验
            def cross(p, q, r):
                return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            
            rect_points = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
            
            for i in range(4):
                p = rect_points[i]
                q = rect_points[(i+1)%4]
                if cross(p, q, p1) * cross(p, q, p2) < 0:
                    return True
            
            # 检查线段是否在矩形内部
            if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or               (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                return True
            
            return False
        
        return line_segment_rect_intersection(start, end, obstacle_pos, obstacle_size)
    
    def _generate_detour_path(self, start, goal, obstacles):
        """
        生成绕过障碍物的路径
        
        Args:
            start: 起始位置
            goal: 目标位置
            obstacles: 障碍物列表
        
        Returns:
            path: 绕过障碍物的路径
        """
        # 简化版本：生成一个绕过障碍物的路径
        path = [start]
        
        # 为每个障碍物生成绕行点
        for obstacle in obstacles:
            obs_pos = obstacle['position'][:2]
            # 生成障碍物右侧的绕行点
            detour_point = [obs_pos[0] + 5.0, obs_pos[1] + 5.0]
            path.append(detour_point)
        
        path.append(goal)
        return path
    
    def _calculate_path_length(self, path):
        """
        计算路径长度
        
        Args:
            path: 路径点列表
        
        Returns:
            length: 路径长度
        """
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += np.sqrt(dx*dx + dy*dy)
        
        return length
    
    def _generate_waypoints(self, path):
        """
        生成航点
        
        Args:
            path: 路径点列表
        
        Returns:
            waypoints: 航点列表
        """
        # 简化版本：直接使用路径点作为航点
        waypoints = []
        for point in path:
            waypoints.append([point[0], point[1], 0.0])  # 添加z坐标
        
        return waypoints
    
    def _calculate_path_cost(self, path, obstacles=None, terrain_map=None):
        """
        计算路径成本
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            terrain_map: 地形地图
        
        Returns:
            cost: 路径成本
        """
        # 使用地面力学模型计算路径成本
        if terrain_map:
            # 计算能耗和通过性成本
            cost, cost_details = self.terramechanics.calculate_path_cost(path, terrain_map, obstacles)
            
            # 更新规划结果中的成本详细信息
            self.plan_result['cost_details'] = cost_details
            
            return cost
        else:
            # 简化版本：成本包括路径长度和与障碍物的距离
            cost = 0.0
            
            # 路径长度成本
            length_cost = self._calculate_path_length(path)
            cost += length_cost
            
            # 障碍物距离成本
            if obstacles:
                for point in path:
                    min_distance = float('inf')
                    for obstacle in obstacles:
                        obs_pos = obstacle['position'][:2]
                        dx = point[0] - obs_pos[0]
                        dy = point[1] - obs_pos[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        min_distance = min(min_distance, distance)
                    
                    # 距离障碍物越近，成本越高
                    if min_distance < 5.0:
                        cost += (5.0 - min_distance) * 10
            
            # 地形成本
            terrain_cost = len(path) * 0.1
            cost += terrain_cost
            
            return cost
    
    def validate_path(self, path, obstacles=None, terrain_map=None):
        """
        验证路径
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            terrain_map: 地形地图
        
        Returns:
            valid: 是否有效
            message: 验证消息
        """
        # 检查路径是否为空
        if not path:
            return False, "路径为空"
        
        # 检查路径是否与障碍物碰撞
        if obstacles:
            for i in range(1, len(path)):
                for obstacle in obstacles:
                    obs_pos = obstacle['position'][:2]
                    obs_size = obstacle['size'][:2]
                    
                    if self._line_obstacle_collision(path[i-1], path[i], obs_pos, obs_size):
                        return False, f"路径与障碍物 {obstacle['id']} 碰撞"
        
        # 检查路径长度是否合理
        path_length = self._calculate_path_length(path)
        straight_line_distance = np.sqrt(
            (path[-1][0] - path[0][0])**2 + 
            (path[-1][1] - path[0][1])**2
        )
        
        if path_length > straight_line_distance * 3:
            return False, "路径长度过长"
        
        return True, "路径验证通过"
    
    def get_plan_result(self):
        """
        获取规划结果
        
        Returns:
            plan_result: 规划结果
        """
        return self.plan_result.copy()
    
    def reset(self):
        """
        重置规划系统
        """
        self.plan_result = {
            'status': 'idle',
            'path': [],
            'path_length': 0.0,
            'waypoints': [],
            'estimated_time': 0.0,
            'cost': 0.0,
            'message': '',
        }
        print("路径规划系统重置完成")
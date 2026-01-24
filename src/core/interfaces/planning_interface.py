#!/usr/bin/env python3
"""
规划接口模块
连接感知系统和路径规划系统
"""

import numpy as np

class PlanningInterface:
    """
    规划接口类
    """
    
    def __init__(self):
        """
        初始化规划接口
        """
        # 感知数据
        self.perception_data = {
            'robot_state': {
                'pose': np.eye(4).tolist(),
                'velocity': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0]
            },
            'obstacles': [],
            'terrain_features': [],
            'semantic_info': {},
            'trajectory': [],
            'timestamp': 0.0,
        }
        
        # 规划请求
        self.planning_request = {
            'start_position': [0.0, 0.0, 0.0],
            'goal_position': [10.0, 10.0, 0.0],
            'constraints': {},
            'priority': 'normal',
        }
        
        # 规划结果
        self.planning_result = {
            'status': 'idle',
            'path': [],
            'path_length': 0.0,
            'waypoints': [],
            'estimated_time': 0.0,
            'cost': 0.0,
            'message': '',
        }
        
        print("规划接口初始化完成")
    
    def update_perception_data(self, perception_data):
        """
        更新感知数据
        
        Args:
            perception_data: 感知数据
        """
        self.perception_data.update(perception_data)
        print("感知数据更新完成")
    
    def request_path_planning(self, start_position, goal_position, constraints=None):
        """
        请求路径规划
        
        Args:
            start_position: 起始位置
            goal_position: 目标位置
            constraints: 约束条件
        
        Returns:
            planning_result: 规划结果
        """
        print(f"请求路径规划: 起点={start_position}, 终点={goal_position}")
        
        # 更新规划请求
        self.planning_request = {
            'start_position': start_position,
            'goal_position': goal_position,
            'constraints': constraints or {},
            'priority': 'normal',
        }
        
        # 模拟路径规划过程
        # 在实际应用中，这里会调用路径规划系统
        planning_result = self._simulate_planning()
        
        # 更新规划结果
        self.planning_result = planning_result
        
        print(f"路径规划完成: 状态={planning_result['status']}")
        
        return planning_result
    
    def _simulate_planning(self):
        """
        模拟路径规划
        
        Returns:
            planning_result: 规划结果
        """
        # 模拟路径规划过程
        import time
        time.sleep(0.1)  # 模拟规划时间
        
        # 生成模拟路径
        start = self.planning_request['start_position']
        goal = self.planning_request['goal_position']
        
        # 生成直线路径
        path = [start, goal]
        
        # 计算路径长度
        path_length = np.sqrt(
            (goal[0] - start[0])**2 + 
            (goal[1] - start[1])**2 + 
            (goal[2] - start[2])**2
        )
        
        # 生成航点
        waypoints = [start, goal]
        
        # 计算估计时间
        estimated_time = path_length / 0.1  # 假设速度为0.1 m/s
        
        # 计算成本
        cost = path_length * 10  # 简化的成本计算
        
        # 检查是否有障碍物
        obstacles = self.perception_data.get('obstacles', [])
        if obstacles:
            # 生成绕过障碍物的路径
            path = self._generate_detour_path(start, goal, obstacles)
            waypoints = path
            
            # 重新计算路径长度
            path_length = 0.0
            for i in range(1, len(path)):
                dx = path[i][0] - path[i-1][0]
                dy = path[i][1] - path[i-1][1]
                dz = path[i][2] - path[i-1][2]
                path_length += np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 重新计算估计时间
            estimated_time = path_length / 0.1
            
            # 重新计算成本
            cost = path_length * 10 + len(obstacles) * 100
        
        # 构建规划结果
        planning_result = {
            'status': 'success',
            'path': path,
            'path_length': path_length,
            'waypoints': waypoints,
            'estimated_time': estimated_time,
            'cost': cost,
            'message': '路径规划成功',
        }
        
        return planning_result
    
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
        path = [start]
        
        # 为每个障碍物生成绕行点
        for obstacle in obstacles:
            obs_pos = obstacle['position']
            # 生成障碍物右侧的绕行点
            detour_point = [obs_pos[0] + 5.0, obs_pos[1] + 5.0, 0.0]
            path.append(detour_point)
        
        path.append(goal)
        return path
    
    def get_planning_result(self):
        """
        获取规划结果
        
        Returns:
            planning_result: 规划结果
        """
        return self.planning_result.copy()
    
    def get_perception_data(self):
        """
        获取感知数据
        
        Returns:
            perception_data: 感知数据
        """
        return self.perception_data.copy()
    
    def get_planning_request(self):
        """
        获取规划请求
        
        Returns:
            planning_request: 规划请求
        """
        return self.planning_request.copy()
    
    def reset(self):
        """
        重置规划接口
        """
        # 重置感知数据
        self.perception_data = {
            'robot_state': {
                'pose': np.eye(4).tolist(),
                'velocity': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0]
            },
            'obstacles': [],
            'terrain_features': [],
            'semantic_info': {},
            'trajectory': [],
            'timestamp': 0.0,
        }
        
        # 重置规划请求
        self.planning_request = {
            'start_position': [0.0, 0.0, 0.0],
            'goal_position': [10.0, 10.0, 0.0],
            'constraints': {},
            'priority': 'normal',
        }
        
        # 重置规划结果
        self.planning_result = {
            'status': 'idle',
            'path': [],
            'path_length': 0.0,
            'waypoints': [],
            'estimated_time': 0.0,
            'cost': 0.0,
            'message': '',
        }
        
        print("规划接口重置完成")
    
    def validate_path(self, path):
        """
        验证路径
        
        Args:
            path: 路径点列表
        
        Returns:
            valid: 是否有效
            message: 验证消息
        """
        # 检查路径是否为空
        if not path:
            return False, "路径为空"
        
        # 检查路径是否包含起始点和终点
        if len(path) < 2:
            return False, "路径至少需要包含两个点"
        
        # 检查路径是否与障碍物碰撞
        obstacles = self.perception_data.get('obstacles', [])
        if obstacles:
            for i in range(1, len(path)):
                for obstacle in obstacles:
                    if self._line_obstacle_collision(path[i-1], path[i], obstacle):
                        return False, f"路径与障碍物 {obstacle.get('id', '')} 碰撞"
        
        return True, "路径验证通过"
    
    def _line_obstacle_collision(self, start, end, obstacle):
        """
        检查线段是否与障碍物碰撞
        
        Args:
            start: 线段起点
            end: 线段终点
            obstacle: 障碍物
        
        Returns:
            collision: 是否碰撞
        """
        # 简化的碰撞检测
        obs_pos = obstacle['position']
        obs_size = obstacle.get('size', [1.0, 1.0, 1.0])
        
        # 计算线段到障碍物中心的距离
        def distance_to_line(p, a, b):
            """计算点到线段的距离"""
            pa = np.array(p) - np.array(a)
            ba = np.array(b) - np.array(a)
            t = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
            return np.linalg.norm(pa - t * ba)
        
        distance = distance_to_line(obs_pos, start, end)
        
        # 检查距离是否小于障碍物半径
        obstacle_radius = max(obs_size) / 2
        return distance < obstacle_radius + 0.5  # 0.5是安全margin
    
    def optimize_path(self, path):
        """
        优化路径
        
        Args:
            path: 原始路径
        
        Returns:
            optimized_path: 优化后的路径
        """
        print("优化路径")
        
        # 简化的路径优化
        # 在实际应用中，这里会使用路径优化算法
        optimized_path = self._simplify_path(path)
        
        print(f"路径优化完成: 路径点数量从{len(path)}减少到{len(optimized_path)}")
        
        return optimized_path
    
    def _simplify_path(self, path, epsilon=0.5):
        """
        使用Ramer-Douglas-Peucker算法简化路径
        
        Args:
            path: 原始路径
            epsilon: 容差
        
        Returns:
            simplified_path: 简化后的路径
        """
        if len(path) <= 2:
            return path
        
        # 找到离线段最远的点
        start = np.array(path[0])
        end = np.array(path[-1])
        max_distance = 0
        farthest_point = 1
        
        for i in range(1, len(path)-1):
            point = np.array(path[i])
            # 计算点到线段的距离
            def distance_to_line(p, a, b):
                pa = p - a
                ba = b - a
                t = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
                return np.linalg.norm(pa - t * ba)
            
            distance = distance_to_line(point, start, end)
            if distance > max_distance:
                max_distance = distance
                farthest_point = i
        
        # 如果最远点的距离大于容差，递归简化
        if max_distance > epsilon:
            left = self._simplify_path(path[:farthest_point+1], epsilon)
            right = self._simplify_path(path[farthest_point:], epsilon)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]
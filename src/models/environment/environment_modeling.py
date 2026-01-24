#!/usr/bin/env python3
"""
环境建模模块
基于嫦娥6号数据的月球环境建模
"""

import numpy as np
import matplotlib.pyplot as plt

class EnvironmentModeling:
    """
    月球环境建模类
    """
    
    def __init__(self, map_resolution=0.1, map_size=(50.0, 50.0)):
        """
        初始化环境建模
        
        Args:
            map_resolution: 地图分辨率 (m/像素)
            map_size: 地图大小 (m)
        """
        self.map_resolution = map_resolution
        self.map_size = map_size
        self.map_width = int(map_size[0] / map_resolution)
        self.map_height = int(map_size[1] / map_resolution)
        
        # 初始化地图
        self.elevation_map = np.zeros((self.map_height, self.map_width))
        self.obstacle_map = np.zeros((self.map_height, self.map_width))
        self.traversability_map = np.ones((self.map_height, self.map_width))
        
        # 障碍物列表
        self.obstacles = []
        
        # 地形特征
        self.terrain_features = []
        
        # 更新次数
        self.update_count = 0
        
        print(f"环境建模初始化完成: 分辨率={map_resolution}m, 地图大小={map_size}m")
    
    def update_map(self, sensor_data):
        """
        使用传感器数据更新环境地图
        
        Args:
            sensor_data: 传感器数据，包含点云、语义分割等
        """
        # 提取点云数据
        if 'point_cloud' in sensor_data:
            point_cloud = sensor_data['point_cloud']
            self._process_point_cloud(point_cloud)
        
        # 提取语义分割
        if 'semantic_segmentation' in sensor_data:
            semantic_segmentation = sensor_data['semantic_segmentation']
            self._process_semantic_segmentation(semantic_segmentation)
        
        # 提取地形特征
        if 'terrain_features' in sensor_data:
            terrain_features = sensor_data['terrain_features']
            self._process_terrain_features(terrain_features)
        
        self.update_count += 1
        print(f"地图更新完成，累计更新次数: {self.update_count}")
    
    def _process_point_cloud(self, point_cloud):
        """
        处理点云数据
        
        Args:
            point_cloud: 点云数据 (N, 3)
        """
        # 过滤有效点
        valid_points = point_cloud[point_cloud[:, 2] > -2]  # 过滤地面以下的点
        
        if len(valid_points) > 0:
            # 更新高程图
            for point in valid_points:
                x, y, z = point
                # 转换为地图坐标
                map_x = int((x + self.map_size[0]/2) / self.map_resolution)
                map_y = int((y + self.map_size[1]/2) / self.map_resolution)
                
                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    self.elevation_map[map_y, map_x] = max(self.elevation_map[map_y, map_x], z)
        
        # 检测障碍物
        self._detect_obstacles(valid_points)
    
    def _process_semantic_segmentation(self, semantic_segmentation):
        """
        处理语义分割数据
        
        Args:
            semantic_segmentation: 语义分割结果
        """
        # 这里可以根据语义分割结果更新障碍物地图和可通行性地图
        # 简单示例：将岩石区域标记为障碍物
        rock_regions = np.where(semantic_segmentation == 2)
        if len(rock_regions[0]) > 0:
            print(f"检测到岩石区域: {len(rock_regions[0])} 像素")
    
    def _process_terrain_features(self, terrain_features):
        """
        处理地形特征数据
        
        Args:
            terrain_features: 地形特征数据
        """
        self.terrain_features.append(terrain_features)
        
        # 根据地形特征更新可通行性地图
        if 'roughness' in terrain_features:
            roughness = terrain_features['roughness']
            # 粗糙度越高，可通行性越低
            traversability_factor = max(0, 1 - roughness * 5)
            self.traversability_map *= traversability_factor
    
    def _detect_obstacles(self, point_cloud):
        """
        从点云数据中检测障碍物
        
        Args:
            point_cloud: 点云数据
        """
        # 简单的障碍物检测算法
        # 计算点云的高度分布
        if len(point_cloud) > 0:
            z_values = point_cloud[:, 2]
            mean_z = np.mean(z_values)
            std_z = np.std(z_values)
            
            # 高度异常的点视为障碍物
            obstacle_points = point_cloud[z_values > mean_z + 2 * std_z]
            
            if len(obstacle_points) > 10:  # 至少10个点才视为障碍物
                # 计算障碍物中心
                obstacle_center = np.mean(obstacle_points[:, :2], axis=0)
                obstacle_size = np.max(obstacle_points[:, :2], axis=0) - np.min(obstacle_points[:, :2], axis=0)
                
                # 添加到障碍物列表
                obstacle = {
                    'id': len(self.obstacles) + 1,
                    'position': obstacle_center.tolist(),
                    'size': obstacle_size.tolist(),
                    'height': np.max(obstacle_points[:, 2]) - np.min(obstacle_points[:, 2]),
                    'point_count': len(obstacle_points)
                }
                
                self.obstacles.append(obstacle)
                print(f"检测到障碍物: ID={obstacle['id']}, 位置={obstacle['position']}, 大小={obstacle['size']}")
    
    def visualize_maps(self):
        """
        可视化环境地图
        
        Returns:
            可视化图像
        """
        # 创建可视化图像
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 高程图
        im1 = ax1.imshow(self.elevation_map, cmap='terrain', origin='lower')
        ax1.set_title('高程图')
        fig.colorbar(im1, ax=ax1, label='高度 (m)')
        
        # 障碍物图
        im2 = ax2.imshow(self.obstacle_map, cmap='gray', origin='lower')
        ax2.set_title('障碍物图')
        fig.colorbar(im2, ax=ax2, label='障碍物概率')
        
        # 可通行性图
        im3 = ax3.imshow(self.traversability_map, cmap='viridis', origin='lower')
        ax3.set_title('可通行性图')
        fig.colorbar(im3, ax=ax3, label='可通行性')
        
        plt.tight_layout()
        
        # 转换为数组返回
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        return image_array
    
    def save_map(self, filename):
        """
        保存环境地图
        
        Args:
            filename: 保存文件名
        """
        np.savez(filename,
                 elevation_map=self.elevation_map,
                 obstacle_map=self.obstacle_map,
                 traversability_map=self.traversability_map,
                 obstacles=self.obstacles,
                 terrain_features=self.terrain_features,
                 map_resolution=self.map_resolution,
                 map_size=self.map_size)
        print(f"地图保存完成: {filename}")
    
    def load_map(self, filename):
        """
        加载环境地图
        
        Args:
            filename: 加载文件名
        """
        try:
            data = np.load(filename, allow_pickle=True)
            self.elevation_map = data['elevation_map']
            self.obstacle_map = data['obstacle_map']
            self.traversability_map = data['traversability_map']
            self.obstacles = data['obstacles'].tolist()
            self.terrain_features = data['terrain_features'].tolist()
            self.map_resolution = data['map_resolution']
            self.map_size = data['map_size']
            
            # 更新地图尺寸
            self.map_width = self.elevation_map.shape[1]
            self.map_height = self.elevation_map.shape[0]
            
            print(f"地图加载完成: {filename}")
            return True
        except Exception as e:
            print(f"地图加载失败: {e}")
            return False
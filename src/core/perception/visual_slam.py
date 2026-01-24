#!/usr/bin/env python3
"""
视觉SLAM模块
基于嫦娥6号数据的月球车视觉SLAM
"""

import numpy as np
import cv2

class VisualSLAM:
    """
    视觉SLAM类
    """
    
    def __init__(self, camera_matrix, dist_coeffs):
        """
        初始化视觉SLAM
        
        Args:
            camera_matrix: 相机内参矩阵
            dist_coeffs: 相机畸变系数
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 关键帧
        self.keyframes = []
        
        # 地图点
        self.map_points = []
        
        # 跟踪状态
        self.tracking_state = 'INITIALIZING'
        
        # 当前帧
        self.current_frame = None
        
        # 当前位姿
        self.current_pose = np.eye(4)
        
        # 特征提取器
        self.feature_extractor = cv2.ORB_create(nfeatures=1000)
        
        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        print("视觉SLAM初始化完成")
    
    def track(self, image):
        """
        跟踪当前帧
        
        Args:
            image: 当前图像
        
        Returns:
            success: 跟踪是否成功
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 提取特征点
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        
        # 初始化
        if self.tracking_state == 'INITIALIZING':
            if len(keypoints) > 100:
                # 创建第一个关键帧
                keyframe = {
                    'id': 0,
                    'image': gray,
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'pose': np.eye(4),
                    'timestamp': 0.0,
                }
                self.keyframes.append(keyframe)
                self.tracking_state = 'TRACKING'
                print("视觉SLAM初始化成功，开始跟踪")
                return True
            else:
                return False
        
        # 跟踪
        elif self.tracking_state == 'TRACKING':
            # 与上一关键帧匹配
            prev_keyframe = self.keyframes[-1]
            matches = self.matcher.match(descriptors, prev_keyframe['descriptors'])
            
            # 过滤匹配
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > 20:
                # 计算位姿
                self._estimate_pose(keypoints, prev_keyframe['keypoints'], good_matches)
                
                # 检查是否需要创建新的关键帧
                if self._need_new_keyframe():
                    self._create_keyframe(gray, keypoints, descriptors)
                
                return True
            else:
                self.tracking_state = 'LOST'
                print("视觉SLAM跟踪失败，进入丢失状态")
                return False
        
        # 丢失
        elif self.tracking_state == 'LOST':
            # 尝试重定位
            if self._relocalize(gray, keypoints, descriptors):
                self.tracking_state = 'TRACKING'
                print("视觉SLAM重定位成功，恢复跟踪")
                return True
            else:
                return False
        
        return False
    
    def _estimate_pose(self, keypoints, prev_keypoints, matches):
        """
        估计位姿
        
        Args:
            keypoints: 当前帧特征点
            prev_keypoints: 上一帧特征点
            matches: 匹配
        """
        # 简化版本：假设位姿不变
        self.current_pose = np.eye(4)
    
    def _need_new_keyframe(self):
        """
        判断是否需要创建新的关键帧
        
        Returns:
            need: 是否需要
        """
        # 简化版本：每5帧创建一个关键帧
        return len(self.keyframes) % 5 == 0
    
    def _create_keyframe(self, image, keypoints, descriptors):
        """
        创建新的关键帧
        
        Args:
            image: 当前图像
            keypoints: 特征点
            descriptors: 描述子
        """
        keyframe = {
            'id': len(self.keyframes),
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': self.current_pose.copy(),
            'timestamp': len(self.keyframes),
        }
        self.keyframes.append(keyframe)
        print(f"创建新的关键帧: ID={keyframe['id']}")
    
    def _relocalize(self, image, keypoints, descriptors):
        """
        重定位
        
        Args:
            image: 当前图像
            keypoints: 特征点
            descriptors: 描述子
        
        Returns:
            success: 重定位是否成功
        """
        # 简化版本：尝试与所有关键帧匹配
        for keyframe in self.keyframes:
            matches = self.matcher.match(descriptors, keyframe['descriptors'])
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > 30:
                return True
        
        return False
    
    def get_trajectory(self):
        """
        获取轨迹
        
        Returns:
            trajectory: 轨迹点列表
        """
        trajectory = []
        for keyframe in self.keyframes:
            pose = keyframe['pose']
            trajectory.append([pose[0, 3], pose[1, 3], pose[2, 3]])
        
        return np.array(trajectory)
    
    def get_map_points(self):
        """
        获取地图点
        
        Returns:
            map_points: 地图点列表
        """
        return self.map_points
    
    def optimize_map(self):
        """
        优化地图
        """
        # 简化版本：不做任何优化
        pass
    
    def reset(self):
        """
        重置视觉SLAM
        """
        self.keyframes = []
        self.map_points = []
        self.tracking_state = 'INITIALIZING'
        self.current_frame = None
        self.current_pose = np.eye(4)
        print("视觉SLAM重置完成")
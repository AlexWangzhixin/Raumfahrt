#!/usr/bin/env python3
"""
感知系统性能测试脚本

功能：
1. 测试各个模块的功能
2. 验证系统集成性能
3. 测量计算时间和资源消耗
4. 生成性能报告
"""

import numpy as np
import cv2
import time
import psutil
import os
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入感知系统模块
try:
    from src.thesis_integration.chapter5_visual_slam import VisualSLAM
    from src.perception.semantic_segmentation import SemanticSegmentation, TerrainFeatureExtractor
    from src.perception.sensor_fusion.multi_sensor_fusion import MultiSensorFusion
    from src.models.environment.environment_modeling import EnvironmentModeling
    from src.core.planning_interface import PlanningInterface
    print("成功导入所有感知系统模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    exit(1)

class PerceptionSystemTester:
    """感知系统测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.test_results = []
        self.system_usage = []
        self.test_count = 0
        
        # 测试配置
        self.test_config = {
            'image_size': (480, 640),
            'num_test_frames': 10,
            'map_resolution': 0.1,
            'map_size': (50, 50),
            'camera_matrix': np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ]),
            'dist_coeffs': np.zeros(5)
        }
        
        # 初始化感知系统模块
        self.modules = {}
        self._initialize_modules()
        
        print("感知系统测试器初始化完成")
    
    def _initialize_modules(self):
        """初始化感知系统模块"""
        try:
            # 初始化视觉SLAM
            print("初始化视觉SLAM...")
            self.modules['slam'] = VisualSLAM(
                self.test_config['camera_matrix'],
                self.test_config['dist_coeffs']
            )
            
            # 初始化语义分割
            print("初始化语义分割...")
            self.modules['segmentation'] = SemanticSegmentation()
            
            # 初始化地形特征提取
            print("初始化地形特征提取...")
            self.modules['terrain'] = TerrainFeatureExtractor()
            
            # 初始化多传感器融合
            print("初始化多传感器融合...")
            self.modules['fusion'] = MultiSensorFusion(dt=0.01)
            
            # 初始化环境建模
            print("初始化环境建模...")
            self.modules['environment'] = EnvironmentModeling(
                map_resolution=self.test_config['map_resolution'],
                map_size=self.test_config['map_size']
            )
            
            # 初始化规划接口
            print("初始化规划接口...")
            self.modules['planning'] = PlanningInterface()
            
            print("所有模块初始化成功")
        except Exception as e:
            print(f"模块初始化失败: {e}")
    
    def test_module_performance(self, module_name: str, test_function) -> Dict:
        """
        测试单个模块性能
        
        Args:
            module_name: 模块名称
            test_function: 测试函数
        
        Returns:
            测试结果字典
        """
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 执行测试
        try:
            result = test_function()
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 计算性能指标
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        test_result = {
            'module': module_name,
            'test_number': self.test_count,
            'timestamp': time.time(),
            'success': success,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'error': error,
            'result': result
        }
        
        self.test_results.append(test_result)
        self.test_count += 1
        
        return test_result
    
    def test_slam_performance(self):
        """测试视觉SLAM性能"""
        def test_function():
            # 生成测试图像
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_image, "Test Frame", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 测试跟踪
            success = self.modules['slam'].track(test_image)
            
            # 测试地图优化
            if hasattr(self.modules['slam'], 'keyframes') and len(self.modules['slam'].keyframes) > 2:
                if hasattr(self.modules['slam'], 'optimize'):
                    self.modules['slam'].optimize()
            
            return {
                'tracking_success': success,
                'keyframe_count': len(getattr(self.modules['slam'], 'keyframes', [])),
                'mappoint_count': len(getattr(self.modules['slam'], 'map_points', []))
            }
        
        result = self.test_module_performance('SLAM', test_function)
        print(f"SLAM测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def test_fusion_performance(self):
        """测试多传感器融合性能"""
        def test_function():
            # 生成测试传感器数据
            sensor_data = {
                'camera': {
                    'pose': np.eye(4)
                },
                'imu': {
                    'acceleration': np.array([0.1, 0.0, -1.62]),
                    'angular_velocity': np.array([0.01, 0.02, 0.03])
                },
                'lidar': {
                    'point_cloud': np.random.rand(100, 3)
                },
                'wheel_encoder': {
                    'distance': 0.1,
                    'angle': 0.05
                }
            }
            
            # 测试融合
            self.modules['fusion'].fuse_sensor_data(sensor_data)
            
            # 获取状态
            state = self.modules['fusion'].get_state()
            pose = self.modules['fusion'].get_pose()
            
            return {
                'position': state['position'].tolist(),
                'velocity': state['velocity'].tolist(),
                'orientation': state['orientation'].tolist()
            }
        
        result = self.test_module_performance('Fusion', test_function)
        print(f"多传感器融合测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def test_environment_performance(self):
        """测试环境建模性能"""
        def test_function():
            # 生成测试点云
            num_points = 1000
            point_cloud = np.random.rand(num_points, 3) * 10 - 5
            point_cloud[:, 2] = np.random.rand(num_points) * 2
            
            # 生成测试位姿
            pose = np.eye(4)
            
            # 生成测试数据
            sensor_data = {
                'pose': pose,
                'point_cloud': point_cloud,
                'semantic_segmentation': np.zeros((480, 640), dtype=np.uint8),
                'terrain_features': {
                    'roughness': 0.1,
                    'slope': np.random.rand(480, 640) * 0.5,
                    'curvature': np.random.rand(480, 640) * 0.1
                }
            }
            
            # 测试地图更新
            self.modules['environment'].update_map(sensor_data)
            
            # 获取地图数据
            map_data = self.modules['environment'].get_map_data()
            
            # 测试可视化
            visualization = self.modules['environment'].visualize_maps()
            
            return {
                'occupancy_map_shape': map_data['occupancy_map'].shape,
                'height_map_shape': map_data['height_map'].shape,
                'semantic_map_shape': map_data['semantic_map'].shape,
                'obstacle_count': len(map_data['obstacles'])
            }
        
        result = self.test_module_performance('Environment', test_function)
        print(f"环境建模测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def test_planning_performance(self):
        """测试规划接口性能"""
        def test_function():
            # 生成测试感知数据
            test_perception_data = {
                'robot_state': {
                    'pose': np.eye(4).tolist(),
                    'velocity': [0.1, 0.0, 0.0],
                    'orientation': [0.0, 0.0, 0.0]
                },
                'obstacles': [
                    {
                        'id': 1,
                        'position': [1.0, 1.0, 0.0],
                        'size': [0.5, 0.5, 0.5],
                        'type': 'rock',
                        'confidence': 0.9
                    }
                ],
                'terrain_features': [
                    {
                        'id': 1,
                        'position': [0.5, 0.5, 0.0],
                        'type': 'roughness',
                        'value': 0.2
                    }
                ],
                'semantic_info': {
                    'terrain_type': 'lunar_soil',
                    'lighting_condition': 'daylight'
                },
                'trajectory': [[0, 0, 0], [0.1, 0, 0]],
                'timestamp': time.time()
            }
            
            # 更新感知数据
            self.modules['planning'].update_perception_data(test_perception_data)
            
            # 测试获取规划数据
            planning_data = self.modules['planning'].get_planning_data()
            
            # 测试路径规划请求
            start_position = [0, 0, 0]
            goal_position = [5, 5, 0]
            plan_result = self.modules['planning'].request_path_planning(start_position, goal_position)
            
            # 测试环境信息查询
            environment_info = self.modules['planning'].get_environment_info([1, 1, 0], radius=3.0)
            
            return {
                'planning_data_available': bool(planning_data),
                'plan_success': plan_result['status'] == 'success',
                'waypoint_count': len(plan_result['waypoints']),
                'environment_info_available': bool(environment_info)
            }
        
        result = self.test_module_performance('Planning', test_function)
        print(f"规划接口测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def test_segmentation_performance(self):
        """测试语义分割性能"""
        def test_function():
            # 生成测试图像
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_image, (100, 100), (200, 200), (128, 128, 128), -1)  # 月壤
            cv2.circle(test_image, (400, 240), 50, (255, 0, 0), -1)  # 岩石
            
            # 测试分割
            segmentation_map, semantic_info = self.modules['segmentation'].segment(test_image)
            
            # 测试可视化
            visualization = self.modules['segmentation'].visualize(test_image, segmentation_map)
            
            return {
                'segmentation_shape': segmentation_map.shape,
                'obstacle_count': len(semantic_info['obstacles']),
                'terrain_type_count': len(semantic_info['terrain_types']),
                'class_distribution': semantic_info['class_distribution']
            }
        
        result = self.test_module_performance('Segmentation', test_function)
        print(f"语义分割测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def test_terrain_performance(self):
        """测试地形特征提取性能"""
        def test_function():
            # 生成测试深度图
            depth_map = np.random.rand(480, 640) * 10
            
            # 测试特征提取
            features = self.modules['terrain'].extract_features(depth_map)
            
            # 测试可视化
            visualization = self.modules['terrain'].visualize_features(depth_map, features)
            
            return {
                'roughness': features['roughness'],
                'max_slope': np.max(features['slope']),
                'max_curvature': np.max(features['curvature']),
                'traversability': features['traversability'],
                'height_stats': features['height_stats']
            }
        
        result = self.test_module_performance('Terrain', test_function)
        print(f"地形特征提取测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def test_integration_performance(self):
        """测试系统集成性能"""
        def test_function():
            start_time = time.time()
            
            # 生成测试数据
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            depth_map = np.random.rand(480, 640) * 10
            
            # 1. 视觉SLAM
            slam_success = self.modules['slam'].track(test_image)
            
            # 2. 语义分割
            segmentation_map, semantic_info = self.modules['segmentation'].segment(test_image)
            
            # 3. 地形特征提取
            terrain_features = self.modules['terrain'].extract_features(depth_map)
            
            # 4. 多传感器融合
            sensor_data = {
                'camera': {'pose': np.eye(4)},
                'imu': {'acceleration': [0.1, 0.0, -1.62], 'angular_velocity': [0.01, 0.02, 0.03]},
                'lidar': {'point_cloud': np.random.rand(100, 3)}
            }
            self.modules['fusion'].fuse_sensor_data(sensor_data)
            
            # 5. 环境建模
            env_data = {
                'pose': np.eye(4),
                'point_cloud': np.random.rand(100, 3),
                'semantic_segmentation': segmentation_map,
                'terrain_features': terrain_features
            }
            self.modules['environment'].update_map(env_data)
            
            # 6. 规划接口
            perception_data = {
                'robot_state': {
                    'pose': np.eye(4),
                    'velocity': [0.1, 0.0, 0.0]
                },
                'obstacles': semantic_info.get('obstacles', []),
                'terrain_features': [{
                    'position': [0, 0, 0],
                    'type': 'roughness',
                    'value': terrain_features['roughness']
                }],
                'semantic_info': semantic_info,
                'environment_map': self.modules['environment'].get_map_data(),
                'trajectory': [[0, 0, 0]],
                'timestamp': time.time()
            }
            self.modules['planning'].update_perception_data(perception_data)
            
            # 7. 路径规划
            plan_result = self.modules['planning'].request_path_planning([0, 0, 0], [5, 5, 0])
            
            integration_time = time.time() - start_time
            
            return {
                'slam_success': slam_success,
                'segmentation_success': bool(segmentation_map.size > 0),
                'terrain_success': bool(terrain_features),
                'fusion_success': True,
                'environment_success': True,
                'planning_success': plan_result['status'] == 'success',
                'integration_time': integration_time,
                'obstacle_count': len(semantic_info.get('obstacles', [])),
                'traversability': terrain_features.get('traversability', 0)
            }
        
        result = self.test_module_performance('Integration', test_function)
        print(f"系统集成测试完成，执行时间: {result['execution_time']:.3f}s, 内存使用: {result['memory_usage']:.2f}MB")
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=== 开始感知系统性能测试 ===\n")
        
        # 测试各个模块
        test_results = []
        
        print("1. 测试视觉SLAM...")
        test_results.append(self.test_slam_performance())
        
        print("\n2. 测试语义分割...")
        test_results.append(self.test_segmentation_performance())
        
        print("\n3. 测试地形特征提取...")
        test_results.append(self.test_terrain_performance())
        
        print("\n4. 测试多传感器融合...")
        test_results.append(self.test_fusion_performance())
        
        print("\n5. 测试环境建模...")
        test_results.append(self.test_environment_performance())
        
        print("\n6. 测试规划接口...")
        test_results.append(self.test_planning_performance())
        
        print("\n7. 测试系统集成...")
        test_results.append(self.test_integration_performance())
        
        # 生成性能报告
        self.generate_performance_report(test_results)
        
        print("\n=== 感知系统性能测试完成 ===")
    
    def generate_performance_report(self, test_results: List[Dict]):
        """生成性能报告"""
        report_filename = f"perception_system_performance_{int(time.time())}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=== 感知系统性能测试报告 ===\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试环境: {os.name}\n")
            f.write(f"Python版本: {os.sys.version}\n")
            f.write(f"测试模块数量: {len(test_results)}\n\n")
            
            # 模块性能统计
            f.write("=== 模块性能统计 ===\n\n")
            for result in test_results:
                f.write(f"模块: {result['module']}\n")
                f.write(f"  测试结果: {'成功' if result['success'] else '失败'}\n")
                f.write(f"  执行时间: {result['execution_time']:.3f} 秒\n")
                f.write(f"  内存使用: {result['memory_usage']:.2f} MB\n")
                if not result['success']:
                    f.write(f"  错误信息: {result['error']}\n")
                f.write("\n")
            
            # 整体性能分析
            f.write("=== 整体性能分析 ===\n\n")
            success_count = sum(1 for r in test_results if r['success'])
            total_count = len(test_results)
            success_rate = (success_count / total_count) * 100
            
            avg_execution_time = sum(r['execution_time'] for r in test_results) / total_count
            max_execution_time = max(r['execution_time'] for r in test_results)
            min_execution_time = min(r['execution_time'] for r in test_results)
            
            avg_memory_usage = sum(r['memory_usage'] for r in test_results) / total_count
            max_memory_usage = max(r['memory_usage'] for r in test_results)
            
            f.write(f"测试成功率: {success_rate:.1f}% ({success_count}/{total_count})\n")
            f.write(f"平均执行时间: {avg_execution_time:.3f} 秒\n")
            f.write(f"最大执行时间: {max_execution_time:.3f} 秒\n")
            f.write(f"最小执行时间: {min_execution_time:.3f} 秒\n")
            f.write(f"平均内存使用: {avg_memory_usage:.2f} MB\n")
            f.write(f"最大内存使用: {max_memory_usage:.2f} MB\n\n")
            
            # 系统建议
            f.write("=== 系统建议 ===\n\n")
            if success_rate < 80:
                f.write("警告: 测试成功率较低，建议检查模块初始化和依赖项\n")
            if max_execution_time > 1.0:
                f.write("警告: 某些模块执行时间较长，建议优化算法或考虑硬件加速\n")
            if max_memory_usage > 500:
                f.write("警告: 内存使用较高，建议优化内存管理\n")
            
            f.write("建议: 定期运行性能测试，监控系统性能变化\n")
            f.write("建议: 根据实际硬件环境调整模块参数\n")
            f.write("建议: 考虑使用并行处理提高系统响应速度\n")
        
        print(f"性能报告生成完成: {report_filename}")

if __name__ == "__main__":
    # 创建测试器
    tester = PerceptionSystemTester()
    
    # 运行所有测试
    tester.run_all_tests()
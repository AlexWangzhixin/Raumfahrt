import numpy as np
from typing import Dict, List, Tuple

class MultiSensorFusion:
    """
    多传感器融合模块
    
    功能：
    1. 融合相机、IMU、激光雷达等传感器数据
    2. 提供更准确的状态估计（位置、速度、姿态）
    3. 处理传感器数据的时间同步
    4. 传感器故障检测和容错
    """
    
    def __init__(self, dt: float = 0.01):
        """
        初始化多传感器融合模块
        
        Args:
            dt: 时间步长 (秒)
        """
        self.dt = dt
        
        # 状态向量: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state_dim = 12
        self.measurement_dim = 6  # 位置和姿态测量
        
        # 初始化状态
        self.x = np.zeros(self.state_dim)
        
        # 初始化状态协方差矩阵
        self.P = np.eye(self.state_dim) * 0.1
        
        # 初始化过程噪声协方差矩阵
        self.Q = self._initialize_process_noise()
        
        # 初始化测量噪声协方差矩阵
        self.R = self._initialize_measurement_noise()
        
        # 状态转移矩阵
        self.F = self._initialize_state_transition_matrix()
        
        # 测量矩阵
        self.H = self._initialize_measurement_matrix()
        
        # 传感器数据缓冲区
        self.sensor_buffers = {
            'camera': [],
            'imu': [],
            'lidar': [],
            'wheel_encoder': []
        }
        
        # 传感器权重
        self.sensor_weights = {
            'camera': 0.3,
            'imu': 0.4,
            'lidar': 0.2,
            'wheel_encoder': 0.1
        }
        
        # 传感器状态
        self.sensor_status = {
            'camera': True,
            'imu': True,
            'lidar': True,
            'wheel_encoder': True
        }
        
        print("多传感器融合模块初始化完成")
    
    def _initialize_process_noise(self) -> np.ndarray:
        """初始化过程噪声协方差矩阵"""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # 位置噪声
        Q[0:3, 0:3] = np.eye(3) * 0.01
        
        # 速度噪声
        Q[3:6, 3:6] = np.eye(3) * 0.1
        
        # 姿态噪声
        Q[6:9, 6:9] = np.eye(3) * 0.01
        
        # 角速度噪声
        Q[9:12, 9:12] = np.eye(3) * 0.1
        
        return Q
    
    def _initialize_measurement_noise(self) -> np.ndarray:
        """初始化测量噪声协方差矩阵"""
        R = np.eye(self.measurement_dim)
        
        # 位置测量噪声
        R[0:3, 0:3] = np.eye(3) * 0.01
        
        # 姿态测量噪声
        R[3:6, 3:6] = np.eye(3) * 0.01
        
        return R
    
    def _initialize_state_transition_matrix(self) -> np.ndarray:
        """初始化状态转移矩阵"""
        F = np.eye(self.state_dim)
        
        # 位置对速度的导数
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt
        
        # 姿态对角速度的导数
        F[6, 9] = self.dt
        F[7, 10] = self.dt
        F[8, 11] = self.dt
        
        return F
    
    def _initialize_measurement_matrix(self) -> np.ndarray:
        """初始化测量矩阵"""
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 6] = 1
        H[4, 7] = 1
        H[5, 8] = 1
        return H
    
    def predict(self):
        """预测状态"""
        # 预测状态
        self.x = self.F @ self.x
        
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray, sensor_type: str):
        """
        更新状态
        
        Args:
            measurement: 测量值 [x, y, z, roll, pitch, yaw]
            sensor_type: 传感器类型
        """
        # 根据传感器类型调整测量噪声
        if sensor_type == 'camera':
            R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        elif sensor_type == 'imu':
            R = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        elif sensor_type == 'lidar':
            R = np.diag([0.001, 0.001, 0.001, 0.1, 0.1, 0.1])
        elif sensor_type == 'wheel_encoder':
            R = np.diag([0.05, 0.05, 0.1, 0.1, 0.1, 0.1])
        else:
            R = self.R
        
        # 计算预测测量
        z_pred = self.H @ self.x
        
        # 计算残差
        y = measurement - z_pred
        
        # 计算残差协方差
        S = self.H @ self.P @ self.H.T + R
        
        # 计算卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        
        # 更新协方差
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
    
    def fuse_sensor_data(self, sensor_data: Dict):
        """
        融合传感器数据
        
        Args:
            sensor_data: 传感器数据字典
        """
        # 预测
        self.predict()
        
        # 处理每个传感器的数据
        for sensor_type, data in sensor_data.items():
            if data is not None and self.sensor_status[sensor_type]:
                # 检查数据有效性
                if self._is_valid_data(data, sensor_type):
                    # 添加到缓冲区
                    self.sensor_buffers[sensor_type].append(data)
                    
                    # 如果是IMU数据，直接更新
                    if sensor_type == 'imu':
                        self._process_imu_data(data)
                    # 如果是相机数据，进行视觉SLAM更新
                    elif sensor_type == 'camera':
                        self._process_camera_data(data)
                    # 如果是激光雷达数据，进行点云更新
                    elif sensor_type == 'lidar':
                        self._process_lidar_data(data)
                    # 如果是轮式编码器数据，进行里程计更新
                    elif sensor_type == 'wheel_encoder':
                        self._process_wheel_encoder_data(data)
    
    def _process_imu_data(self, data: Dict):
        """处理IMU数据"""
        # 提取加速度和角速度
        accel = data.get('acceleration', np.zeros(3))
        gyro = data.get('angular_velocity', np.zeros(3))
        
        # 计算姿态更新
        # 这里使用简化的姿态更新，实际应该使用四元数或旋转矩阵
        pass
    
    def _process_camera_data(self, data: Dict):
        """处理相机数据"""
        # 提取SLAM结果
        pose = data.get('pose', np.eye(4))
        
        # 转换为测量值
        measurement = np.array([
            pose[0, 3],  # x
            pose[1, 3],  # y
            pose[2, 3],  # z
            0, 0, 0      # 姿态角（需要从旋转矩阵转换）
        ])
        
        # 更新
        self.update(measurement, 'camera')
    
    def _process_lidar_data(self, data: Dict):
        """处理激光雷达数据"""
        # 提取点云数据
        point_cloud = data.get('point_cloud', np.zeros((0, 3)))
        
        # 计算地面高度和障碍物
        # 这里使用简化的处理，实际应该进行地面分割和障碍物检测
        pass
    
    def _process_wheel_encoder_data(self, data: Dict):
        """处理轮式编码器数据"""
        # 提取里程计数据
        distance = data.get('distance', 0.0)
        angle = data.get('angle', 0.0)
        
        # 计算位置更新
        # 这里使用简化的计算，实际应该考虑车轮半径和运动学模型
        pass
    
    def _is_valid_data(self, data: Dict, sensor_type: str) -> bool:
        """检查数据有效性"""
        if data is None:
            return False
        
        # 根据传感器类型检查数据
        if sensor_type == 'imu':
            return 'acceleration' in data and 'angular_velocity' in data
        elif sensor_type == 'camera':
            return 'pose' in data
        elif sensor_type == 'lidar':
            return 'point_cloud' in data
        elif sensor_type == 'wheel_encoder':
            return 'distance' in data and 'angle' in data
        
        return False
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        state = self.x
        
        return {
            'position': state[0:3],
            'velocity': state[3:6],
            'orientation': state[6:9],
            'angular_velocity': state[9:12],
            'covariance': self.P
        }
    
    def get_pose(self) -> np.ndarray:
        """获取当前位姿"""
        state = self.x
        
        # 创建4x4位姿矩阵
        pose = np.eye(4)
        pose[0:3, 3] = state[0:3]
        
        # 从欧拉角创建旋转矩阵
        roll, pitch, yaw = state[6:9]
        R = self._euler_to_rotation_matrix(roll, pitch, yaw)
        pose[0:3, 0:3] = R
        
        return pose
    
    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角转旋转矩阵"""
        # 滚转（x轴）
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 俯仰（y轴）
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 偏航（z轴）
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵
        R = R_z @ R_y @ R_x
        
        return R
    
    def detect_sensor_failure(self):
        """检测传感器故障"""
        # 检查传感器数据一致性
        # 这里使用简化的检测方法，实际应该使用更复杂的故障检测算法
        pass
    
    def visualize_fusion_result(self, trajectory: List[np.ndarray]) -> np.ndarray:
        """可视化融合结果"""
        # 创建轨迹可视化
        # 这里返回一个空数组，实际应该创建一个轨迹图
        return np.array([])

# 测试代码
if __name__ == "__main__":
    # 创建多传感器融合模块
    fusion = MultiSensorFusion(dt=0.01)
    print("多传感器融合模块创建成功")
    
    # 测试状态预测
    print("\n=== 测试状态预测 ===")
    fusion.predict()
    state = fusion.get_state()
    print(f"预测状态 - 位置: {state['position']}")
    print(f"预测状态 - 速度: {state['velocity']}")
    print(f"预测状态 - 姿态: {state['orientation']}")
    
    # 测试传感器数据融合
    print("\n=== 测试传感器数据融合 ===")
    sensor_data = {
        'camera': {
            'pose': np.array([
                [1, 0, 0, 1.0],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1]
            ])
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
    
    fusion.fuse_sensor_data(sensor_data)
    state = fusion.get_state()
    print(f"融合后状态 - 位置: {state['position']}")
    print(f"融合后状态 - 速度: {state['velocity']}")
    print(f"融合后状态 - 姿态: {state['orientation']}")
    
    # 测试位姿获取
    print("\n=== 测试位姿获取 ===")
    pose = fusion.get_pose()
    print(f"位姿矩阵:\n{pose}")
    
    print("\n多传感器融合模块测试完成")
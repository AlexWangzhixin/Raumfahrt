# 月面巡视器仿真配置文件

class Config:
    """
    【全局配置】月面巡视器动力学仿真系统配置
    包含第三章地形建模和第四章动力学仿真的所有参数
    """
    
    # ==========================================
    # 第三章：月面地形建模配置
    # ==========================================
    class Terrain:
        DTM_PATH = "NAC_DTM_CHANGE4.tiff"  # 数字地形模型路径
        RESOLUTION = 0.5  # 地形分辨率 (m/pixel)
        SIZE = 20.0  # 地形尺寸 (m)
        SMOOTHING = True  # 是否平滑地形
        
        # 月面物理参数
        LATERAL_FRICTION = 0.6  # 横向摩擦系数
        SPINNING_FRICTION = 0.1  # 旋转摩擦系数
        ROLLING_FRICTION = 0.05  # 滚动摩擦系数
        RESTITUTION = 0.05  # 弹性恢复系数
    
    # ==========================================
    # 第四章：月球车动力学仿真配置
    # ==========================================
    class Rover:
        URDF_PATH = "r2d2.urdf"  # 巡视器URDF模型路径
        MASS = 150.0  # 巡视器质量 (kg)
        
        # 轮子配置，中小型月球车的典型尺寸
        WHEEL_JOINTS = [2, 3, 6, 7]  # 轮子关节索引
        WHEEL_BASE = 0.5  # 轮距 (m)
        WHEEL_RADIUS = 0.15  # 轮径 (m)
        MAX_FORCE = 100.0  # 电机最大扭矩 (N·m)
        
        # 物理参数，来自PyBullet 物理引擎的推荐设置和月面环境特性
        BODY_LATERAL_FRICTION = 0.8
        BODY_SPINNING_FRICTION = 0.2
        BODY_ROLLING_FRICTION = 0.1
        
        WHEEL_LATERAL_FRICTION = 0.9
        WHEEL_SPINNING_FRICTION = 0.1
        WHEEL_ROLLING_FRICTION = 0.05
        WHEEL_RESTITUTION = 0.01
    
    # ==========================================
    # 仿真环境配置
    # ==========================================
    class Simulation:
        GRAVITY = -1.625  # 月球重力加速度 (m/s²)
        SIMULATION_STEP = 0.01  # 仿真步长 (s)
        PHYSICS_STEPS = 10  # 每步执行的物理仿真次数
        GUI = True  # 是否显示GUI
        
        # 初始状态
        INITIAL_POSITION = [0, 0, 0.5]  # 初始位置 (x, y, z)
        INITIAL_ORIENTATION = [0, 0, 0]  # 初始姿态 (roll, pitch, yaw)
    
    # ==========================================
    # 数据记录配置
    # ==========================================
    class Data:
        LOG_INTERVAL = 0.1  # 日志记录间隔 (s)
        EXPERIMENT_FILE = "lunar_rover_experiment.csv"  # 实验数据保存路径
        TRAJECTORY_PLOT = "rover_trajectory.png"  # 轨迹可视化保存路径
        TERRAIN_PLOT = "lunar_terrain.png"  # 地形可视化保存路径

# 创建全局配置实例
config = Config()
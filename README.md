# 月球车环境与动力学建模系统

## 项目概述

本项目是一个月球车环境与动力学建模系统，主要包含环境建模、动力学建模和多传感器融合等核心模块。系统采用模块化设计，便于扩展和维护。

## 目录结构

```
Raumfahrt/
├── core/                  # 核心功能模块
│   ├── simulation/        # 仿真模块
│   └── visualization/     # 可视化模块
├── data/                  # 数据目录
│   ├── images/            # 图片资源
│   └── visualizations/    # 可视化结果
│       └── dynamics/      # 动力学可视化
├── models/                # 建模模块
│   ├── environment/       # 环境建模
│   │   ├── __init__.py
│   │   └── environment_modeling.py
│   └── dynamics/          # 动力学建模
│       ├── __init__.py
│       ├── lunar_rover_dynamics.py
│       └── dynamics_perception_integration.py
├── perception/            # 感知系统
│   └── sensor_fusion/     # 多传感器融合
│       ├── __init__.py
│       └── multi_sensor_fusion.py
├── utils/                 # 工具函数
├── config/                # 配置文件
├── docs/                  # 文档
│   ├── reports/           # 报告
│   └── plan.md            # 计划文档
├── PathPlanningLunarRovers/ # 路径规划模块
├── requirements.txt       # 依赖项
└── README.md              # 项目说明
```

## 各模块功能说明

### 1. 环境建模 (models/environment/)

**主要功能**：
- 构建栅格地图（2D占用网格）
- 生成高度图（3D地形信息）
- 构建语义地图（带有语义标签的环境表示）
- 地图更新和维护
- 地图压缩和存储优化
- 可通行性分析

**核心文件**：
- `environment_modeling.py`：环境建模主模块，包含`EnvironmentModeling`类

### 2. 动力学建模 (models/dynamics/)

**主要功能**：
- 完整的八轮月球车动力学系统
- 轮地接触力学计算
- 地形交互建模
- 能量消耗计算
- 轨迹预测
- 与感知系统的集成

**核心文件**：
- `lunar_rover_dynamics.py`：月球车动力学模型，包含`LunarRoverDynamics`类
- `dynamics_perception_integration.py`：动力学与感知系统集成，包含`DynamicsPerceptionIntegration`类

### 3. 多传感器融合 (perception/sensor_fusion/)

**主要功能**：
- 融合相机、IMU、激光雷达等传感器数据
- 提供更准确的状态估计（位置、速度、姿态）
- 处理传感器数据的时间同步
- 传感器故障检测和容错

**核心文件**：
- `multi_sensor_fusion.py`：多传感器融合模块，包含`MultiSensorFusion`类

### 4. 路径规划 (PathPlanningLunarRovers/)

**主要功能**：
- A*路径规划算法
- D3QN强化学习路径规划
- 月球环境建模
- 感知系统集成

## 环境和动力学建模位置

### 环境建模
**位置**：`models/environment/environment_modeling.py`
**核心类**：`EnvironmentModeling`
**主要功能**：构建和维护月球表面环境地图，包括占用地图、高度图、语义地图和可通行性地图。

### 动力学建模
**位置**：`models/dynamics/`
**核心类**：
- `LunarRoverDynamics` (`lunar_rover_dynamics.py`)：完整的月球车动力学仿真
- `DynamicsPerceptionIntegration` (`dynamics_perception_integration.py`)：动力学与感知系统集成

## 如何运行项目

1. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行环境建模示例：
   ```bash
   python models/environment/environment_modeling.py
   ```

3. 运行动力学建模示例：
   ```bash
   python models/dynamics/lunar_rover_dynamics.py
   ```

4. 运行多传感器融合示例：
   ```bash
   python perception/sensor_fusion/multi_sensor_fusion.py
   ```

## 技术特点

1. **模块化设计**：各模块独立封装，便于扩展和维护
2. **物理完整性**：完整的牛顿力学模型，考虑所有力和力矩
3. **地形适应性**：基于实际地形模型的交互
4. **感知集成**：为感知系统提供丰富的动力学特征
5. **预测能力**：基于当前控制的轨迹预测
6. **多传感器融合**：提高状态估计的准确性和可靠性

## 未来改进方向

1. **高保真地形模型**：集成真实的月球地形数据
2. **多体动力学**：考虑悬架系统和车轮独立运动
3. **传感器融合**：与实际传感器数据的融合
4. **学习增强**：基于数据驱动的模型增强
5. **实时优化**：计算效率优化，支持实时应用
6. **故障仿真**：车轮故障和地形意外情况的仿真

## 参考文献

1. Bekker, M. G. (1969). Introduction to Terrain-Vehicle Systems. University of Michigan Press.
2. Wong, J. Y. (2001). Theory of Ground Vehicles. John Wiley & Sons.
3. Iagnemma, K., & Dubowsky, S. (2004). Mobile Robots in Rough Terrain: Estimation, Motion Planning, and Control With Application to Planetary Rovers. Kluwer Academic Publishers.
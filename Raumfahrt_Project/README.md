# 月球车导航系统 (Lunar Rover Navigation System)

## 项目简介

本项目是一个面向博士论文的月球车导航系统，基于嫦娥6号数据，实现了完整的环境建模、动力学仿真和路径规划功能。项目采用标准的Python科研项目结构，旨在为月球车的自主导航提供理论支持和算法验证。

## 项目结构

项目采用分层架构设计，严格遵循"环境(Ch3) -> 动力学(Ch4) -> 规划(Ch5)"的逻辑闭环：

```
Raumfahrt_Project/
├── data/                       # [数据层] 存放所有输入数据（不上传git）
│   ├── datasets/               # 原始数据集 (如嫦娥6号数据、分割数据集)
│   ├── models/                 # 预训练模型权重 (.pth, .pt)
│   └── maps/                   # 预设的地图文件 (.npz)
│
├── docs/                       # [文档层] 论文、笔记、API说明
│   ├── references/             # 参考文献 (已发表论文PDF放这里)
│   ├── thesis/                 # 博士论文各章节草稿 (Drafts)
│   ├── plan/                   # 完善计划、答辩问题反馈
│   └── api/                    # 代码接口文档
│
├── outputs/                    # [产出层] 所有运行生成的图表、日志 (git ignore)
│   ├── logs/                   # 训练日志
│   ├── visualizations/         # 动力学对比图、路径规划图
│   └── checkpoints/            # 训练过程中保存的模型检查点
│
├── src/                        # [核心代码层] (Source Code) - 论文核心逻辑
│   ├── __init__.py
│   ├── config/                 # 配置模块
│   │   ├── __init__.py
│   │   └── config.py           # 全局参数 (物理常数、仿真步长)
│   │
│   ├── core/                   # 核心工具库
│   │   ├── __init__.py
│   │   ├── utils.py            # 通用工具函数
│   │   ├── interfaces.py       # 定义抽象基类 (Interface)
│   │   └── visualization.py    # 统一绘图工具类
│   │
│   ├── environment/            # [第三章] 环境建模与数字孪生基座
│   │   ├── __init__.py
│   │   ├── modeling.py         # EnvironmentModeling 类 (高程/语义)
│   │   ├── terramechanics.py   # [核心] Terramechanics 类 (Bekker/Wong公式)
│   │   └── soil_db.py          # 土壤物理参数数据库
│   │
│   ├── dynamics/               # [第四章] 动力学与参数辨识
│   │   ├── __init__.py
│   │   ├── rover_dynamics.py   # LunarRoverDynamics 类 (受力分析)
│   │   └── estimator.py        # [核心] ParameterEstimator 类 (RLS/EKF参数辨识)
│   │
│   ├── perception/             # [第三章/辅助] 感知系统
│   │   ├── __init__.py
│   │   └── segmentation/       # 图像分割相关代码
│   │
│   └── planning/               # [第五章] 路径规划
│       ├── __init__.py
│       ├── global_planner/     # A* 算法
│       │   └── astar.py
│       └── local_planner/      # D3QN / RL 算法
│           ├── agent.py        # D3QNAgent
│           ├── network.py      # 网络结构
│           └── replay_buffer.py
│
├── scripts/                    # [执行层] 启动脚本 (入口)
│   ├── train_agent.py          # 启动RL训练
│   ├── run_digital_twin.py     # 启动动力学孪生仿真
│   ├── run_demo.py             # 系统演示
│   └── analyze_results.py      # 画图/数据分析脚本
│
├── tests/                      # [测试层] 单元测试
│   ├── test_dynamics.py
│   ├── test_env.py
│   └── ...
│
├── .gitignore                  # Git忽略文件
├── README.md                   # 项目说明
└── requirements.txt            # 依赖包列表
```

## 核心功能

### 1. 环境建模 (Chapter 3)
- **地形建模**：基于高程数据的三维地形重建
- **地面力学**：Bekker-Wong轮壤交互模型
- **土壤数据库**：月球表面不同区域的土壤物理参数

### 2. 动力学仿真 (Chapter 4)
- **多体动力学**：六轮月球车的完整动力学模型
- **参数辨识**：基于RLS和EKF的实时参数估计
- **数字孪生**：固定参数与自适应参数的性能对比

### 3. 路径规划 (Chapter 5)
- **全局规划**：基于A*算法的最优路径搜索
- **局部规划**：基于D3QN的实时避障
- **混合规划**：结合全局路径和局部感知的导航策略

## 快速开始

### 环境配置

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd Raumfahrt_Project
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 运行示例

#### 0. 构建第三章环境地图
```bash
python scripts/experiments/ch3_build_environment.py --config configs/ch3_environment.yaml
```
输出目录：`outputs/runs/ch3_environment/<run_id>/environment/environment_maps.npz`

#### 0.1 运行感知特征提取（Perception）
```bash
python scripts/experiments/perception_extract_features.py --config configs/perception.yaml
```
输出目录：`outputs/runs/perception/<run_id>/perception/perception_features.npz`

#### 0.2 运行第四章动力学仿真
```bash
python scripts/experiments/ch4_run_dynamics.py --config configs/ch4_dynamics.yaml
```
输出目录：`outputs/runs/ch4_dynamics/<run_id>/dynamics/dynamics_results.npz`

#### 0.3 运行第五章全局规划
```bash
python scripts/experiments/ch5_run_planning.py --config configs/ch5_planning.yaml
```
输出目录：`outputs/runs/ch5_planning/<run_id>/planning/planning_path.npz`

#### 0.4 运行端到端闭环
```bash
python scripts/run_end_to_end.py --config configs/end_to_end.yaml
```
输出目录：`outputs/runs/end_to_end/<run_id>/summary.json`

#### 1. 运行数字孪生仿真
```bash
python scripts/run_digital_twin.py
```

#### 2. 训练路径规划智能体
```bash
python scripts/train_agent.py
```

#### 3. 运行系统演示
```bash
python scripts/run_demo.py
```

#### 4. 分析仿真结果
```bash
python scripts/analyze_results.py
```

## 关键模块说明

### 环境建模模块 (`src/environment/`)
- **modeling.py**：环境建模主类，负责地形生成和管理
- **terramechanics.py**：地面力学核心实现，包含Bekker公式和Wong-Reece模型
- **soil_db.py**：土壤参数数据库，存储不同月球区域的物理特性

### 动力学模块 (`src/dynamics/`)
- **rover_dynamics.py**：月球车动力学主类，实现六轮驱动的完整动力学模型
- **estimator.py**：参数估计器，使用RLS和EKF算法实时估计地面力学参数

### 规划模块 (`src/planning/`)
- **global_planner/astar.py**：A*算法实现，用于全局路径规划
- **local_planner/agent.py**：D3QN智能体，用于局部避障和路径优化
- **local_planner/network.py**：D3QN网络结构，包含价值网络和策略网络

### 工具模块 (`src/core/`)
- **utils.py**：通用工具函数，包含文件操作、数学计算等
- **interfaces.py**：抽象基类定义，为各模块提供统一接口
- **visualization.py**：可视化工具，用于绘制轨迹、地图和性能曲线

## 测试与验证

项目包含完整的单元测试套件，位于 `tests/` 目录：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_dynamics.py
```

## 文档

### 论文相关
- **docs/thesis/**：博士论文各章节草稿
- **docs/plan/**：完善计划和答辩问题反馈

### 实验说明
- **docs/EXPERIMENTS.md**：可复现实验流程与端到端运行说明

### 代码文档
- **docs/api/**：自动生成的API文档

### 参考文献
- **docs/references/**：相关论文和技术资料

## 依赖项

主要依赖包（详细列表见 `requirements.txt`）：

- **numpy**：数值计算
- **matplotlib**：数据可视化
- **torch**：深度学习框架（用于D3QN）
- **scipy**：科学计算
- **pyyaml**：配置文件解析

## 许可证

本项目仅供学术研究使用，未经授权不得用于商业目的。

## 联系方式

- **作者**：[Your Name]
- **邮箱**：[your.email@university.edu]
- **机构**：[Your University/Institute]

---

*本项目基于嫦娥6号数据，旨在为月球车的自主导航提供理论支持和算法验证。*  
*© 2026 [Your Name] - All Rights Reserved.*

# A*-D3QN-Opt 月球车路径规划系统

基于数字孪生技术的月球车自主导航框架，结合A*全局路径规划与D3QN深度强化学习的局部避障策略。

## 项目简介

本项目实现了一个完整的月球车路径规划系统，用于在动态月球环境中进行自主导航。系统采用分层架构：
- **全局规划层**：使用A*算法规划从起点到目标的最优路径
- **局部避障层**：使用D3QN（Dueling Double DQN）深度强化学习实时避障
- **优先经验回放**：提高训练效率，加速收敛

### 主要特性

- 数字孪生月球环境仿真（地形、障碍物、物理动力学）
- 三种复杂度场景：
  - **Stage 1**：无障碍物环境
  - **Stage 2**：静态障碍物环境
  - **Stage 3**：动态障碍物环境
- 完整的深度强化学习训练流程
- 可视化训练过程和测试结果
- 多方法对比实验支持

---

## 系统要求

### 硬件要求
- CPU：多核处理器（推荐4核以上）
- 内存：至少8GB RAM（推荐16GB）
- GPU：可选，支持CUDA的NVIDIA显卡（推荐用于加速训练）

### 软件要求
- Python 3.8 或更高版本
- 操作系统：Windows / macOS / Linux

---

## 安装步骤

### 1. 克隆或下载项目

```bash
cd PathPlanningLunarRovers
```

### 2. 创建虚拟环境（推荐）

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import torch; print(torch.__version__)"
```

---

## 快速开始

### 训练模型

#### Stage 1 - 无障碍物环境
```bash
python main.py train --method a_star_d3qn_opt --stage 1 --episodes 1000
```

#### Stage 2 - 静态障碍物环境
```bash
python main.py train --method a_star_d3qn_opt --stage 2 --episodes 2000 --render
```

#### Stage 3 - 动态障碍物环境
```bash
python main.py train --method a_star_d3qn_opt --stage 3 --episodes 3000
```

### 测试模型

```bash
python main.py test --model ./results/best_model.pth --stage 2 --episodes 100 --render
```

### 演示模式

```bash
python main.py demo --stage 2 --model ./results/best_model.pth
```

### 方法对比实验

```bash
python main.py compare --episodes 500
```

---

## 命令行参数说明

### 训练参数 (train)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | a_star_d3qn_opt | 训练方法：d3qn, d3qn_per, a_star_d3qn_opt |
| `--stage` | int | 1 | 环境复杂度：1（无障碍）, 2（静态障碍）, 3（动态障碍） |
| `--episodes` | int | 1000 | 训练回合数 |
| `--render` | flag | False | 是否实时渲染训练过程 |
| `--seed` | int | 42 | 随机种子 |
| `--save_dir` | str | ./results | 结果保存目录 |
| `--load_model` | str | None | 加载预训练模型路径 |

### 测试参数 (test)

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--model` | str | 是 | 模型文件路径 |
| `--stage` | int | 否 | 测试环境阶段 |
| `--episodes` | int | 否 | 测试回合数 |
| `--render` | flag | 否 | 是否渲染 |
| `--record_path` | flag | 否 | 是否记录路径轨迹 |

---

## 项目结构

```
PathPlanningLunarRovers/
│
├── main.py                    # 主程序入口
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
├── config.py                  # 配置参数
├── requirements.txt           # 依赖列表
│
├── lunar_environment.py       # 月球环境仿真
├── d3qn_agent.py             # D3QN智能体
├── d3qn_network.py           # D3QN神经网络
├── astar.py                  # A*路径规划
├── perception.py             # 感知模块（RGB-D相机仿真）
├── prioritized_replay.py     # 优先经验回放
├── utils.py                  # 工具函数
│
├── results/                  # 训练结果保存目录
│   ├── models/              # 保存的模型文件
│   ├── logs/                # 训练日志
│   └── plots/               # 可视化图表
│
└── README.md                 # 本文档
```

---

## 核心模块介绍

### 1. 月球环境模块 (lunar_environment.py)

实现数字孪生月球环境，包括：
- **地形模型**：模拟月球表面地形噪声
- **障碍物系统**：支持静态和动态障碍物
- **月球车动力学**：八轮月球车的运动学和动力学模型
- **物理仿真**：月球重力（1.62 m/s²）、摩擦力等

**主要类：**
- `LunarEnvironment`: 月球环境主类
- `LunarRover`: 月球车模型
- `Obstacle`: 障碍物类

**关键方法：**
```python
env = LunarEnvironment(stage=2, render=True)
state = env.reset()                          # 重置环境
state, reward, done, info = env.step(action) # 执行动作
env.render()                                 # 渲染可视化
```

### 2. D3QN智能体 (d3qn_agent.py)

实现Dueling Double DQN算法，负责局部避障决策。

**核心功能：**
- ε-greedy探索策略
- 经验回放缓冲区
- 目标网络软更新
- 优先经验回放（可选）
- A*全局规划集成（可选）

**主要方法：**
```python
agent = D3QNAgent(use_per=True, use_astar=True)
action = agent.select_action(state, features, training=True)
agent.train()                                # 训练网络
agent.save_model(path)                       # 保存模型
```

### 3. D3QN网络 (d3qn_network.py)

实现Dueling架构的深度Q网络。

**网络结构：**
```
Input: 深度图 (4 × 80 × 64) + 附加特征 (8维)
    ↓
CNN特征提取（3层卷积）
    ↓
    ├─→ Value Stream (状态价值)
    └─→ Advantage Stream (动作优势)
    ↓
合并 → Q值输出 (9个动作)
```

**关键组件：**
- 3层卷积网络提取视觉特征
- Dueling架构分离状态价值和动作优势
- 全连接层融合多模态特征

### 4. A*路径规划 (astar.py)

实现A*算法的全局路径规划。

**功能：**
- 栅格地图构建
- 障碍物膨胀处理
- 8方向搜索
- 曼哈顿/欧几里得启发式函数

**使用示例：**
```python
from astar import create_grid_map, AStar

# 创建栅格地图
grid_map = create_grid_map(width, height, resolution, obstacles)

# A*规划
planner = AStar(grid_map, resolution)
path = planner.plan(start, goal)
```

### 5. 感知模块 (perception.py)

模拟RGB-D相机，生成深度图。

**特性：**
- 仿真RGB-D相机（640×480分辨率）
- 深度图预处理（下采样到80×64）
- 帧堆叠（4帧历史）
- 障碍物深度估计

### 6. 优先经验回放 (prioritized_replay.py)

实现基于TD误差的优先经验回放。

**核心算法：**
- SumTree数据结构，O(log n)采样
- 优先级 = |TD误差| + ε
- 重要性采样权重校正

---

## 配置参数说明 (config.py)

### 环境参数
```python
ENV_WIDTH = 10.0              # 环境宽度 (m)
ENV_HEIGHT = 10.0             # 环境高度 (m)
GRID_RESOLUTION = 0.1         # 栅格分辨率 (m)
LUNAR_GRAVITY = 1.62          # 月球重力 (m/s²)
```

### 月球车参数
```python
ROVER_MASS = 50.0             # 质量 (kg)
MAX_VELOCITY = 0.4            # 最大速度 (m/s)
ROVER_COLLISION_RADIUS = 0.36 # 碰撞半径 (m)
```

### 训练参数
```python
LEARNING_RATE = 0.0001        # 学习率
GAMMA = 0.99                  # 折扣因子
BATCH_SIZE = 64               # 批量大小
REPLAY_BUFFER_SIZE = 200000   # 回放缓冲区大小
EPSILON_START = 1.0           # 初始探索率
EPSILON_MIN = 0.10            # 最小探索率
```

### 奖励函数
```python
REWARD_GOAL_REACHED = 500.0              # 到达目标
REWARD_COLLISION = -200.0                # 碰撞惩罚
REWARD_PROGRESS_WEIGHT = 30.0            # 进度奖励系数
REWARD_OBSTACLE_PROXIMITY_WEIGHT = -10.0 # 接近障碍物惩罚
```

---

## 训练流程说明

### 1. 分阶段训练策略（推荐）

**Stage 1（1000回合）→ Stage 2（2000回合）→ Stage 3（3000回合）**

```bash
# Step 1: 无障碍物训练
python main.py train --method a_star_d3qn_opt --stage 1 --episodes 1000

# Step 2: 加载Stage 1模型，继续Stage 2训练
python main.py train --method a_star_d3qn_opt --stage 2 --episodes 2000 \
    --load_model ./results/best_model.pth

# Step 3: 加载Stage 2模型，继续Stage 3训练
python main.py train --method a_star_d3qn_opt --stage 3 --episodes 3000 \
    --load_model ./results/best_model.pth
```

### 2. 训练输出

训练过程中会显示：
```
Episode 100/1000 | Reward: 450.23 | Success: 78% | Collision: 12% | ε: 0.85
Episode 200/1000 | Reward: 512.67 | Success: 85% | Collision: 8%  | ε: 0.72
...
```

### 3. 保存的文件

```
results/
├── best_model.pth              # 最佳模型
├── final_model.pth             # 最终模型
├── training_curve.png          # 训练曲线
├── training_log.txt            # 训练日志
└── checkpoints/
    ├── model_ep100.pth
    ├── model_ep200.pth
    └── ...
```

---

## 测试与评估

### 运行测试

```bash
python main.py test --model ./results/best_model.pth --stage 2 \
    --episodes 100 --render --record_path
```

### 评估指标

测试完成后会输出：
```
============ 测试结果 ============
总回合数: 100
成功率:   92.0%
碰撞率:   6.0%
超时率:   2.0%
平均奖励: 523.45
平均步数: 187.3
平均能量: 1245.8 J
================================
```

### 可视化结果

生成的文件：
- `test_results/statistics.png` - 统计图表
- `test_results/trajectories/` - 路径轨迹图
- `test_results/test_log.txt` - 详细日志

---

## 方法对比

系统支持三种方法对比：

1. **D3QN** - 基础Dueling Double DQN
2. **D3QN-PER** - D3QN + 优先经验回放
3. **A*-D3QN-Opt** - A*全局规划 + D3QN局部避障 + PER（本文方法）

运行对比实验：
```bash
python main.py compare --episodes 500
```

生成对比图：
- 学习曲线对比
- 成功率对比
- 碰撞率对比
- 平均奖励对比

---

## 动作空间

系统定义了9个离散动作：

| 动作ID | 速度 (m/s) | 转向角 (°) | 说明 |
|--------|-----------|-----------|------|
| 0 | 0.2 | +30° | 低速左转 |
| 1 | 0.2 | +15° | 低速小左转 |
| 2 | 0.2 | 0° | 低速直行 |
| 3 | 0.2 | -15° | 低速小右转 |
| 4 | 0.2 | -30° | 低速右转 |
| 5 | 0.4 | +15° | 高速小左转 |
| 6 | 0.4 | 0° | 高速直行 |
| 7 | 0.4 | -15° | 高速小右转 |
| 8 | 0.0 | 0° | 紧急停止 |

---

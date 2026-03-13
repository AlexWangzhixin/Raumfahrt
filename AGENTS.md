# AGENTS.md - 项目指南

本文档为 AI 编程助手提供项目背景、架构和开发规范说明。

---

## 项目概述

**Raumfahrt**（德语"航天"之意）是一个面向博士论文的月球车导航系统研究项目，基于嫦娥6号数据，实现完整的环境建模、动力学仿真和路径规划功能。

### 核心研究内容

项目按论文章节组织，遵循"环境(Ch3) -> 动力学(Ch4) -> 规划(Ch5)"的逻辑闭环：

- **第三章：环境建模** - 月面地形数字孪生建模、月壤环境建模（基于Bekker-Wong理论）
- **第四章：动力学** - 六轮月球车多体动力学、基于RLS/EKF的参数辨识
- **第五章：规划** - A*全局规划 + D3QN（Dueling Double DQN）局部避障的混合算法

---

## 技术栈

### 编程语言与依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 主开发语言 |
| numpy | >=2.0 | 数值计算 |
| scipy | >=1.13 | 科学计算 |
| matplotlib | >=3.9 | 数据可视化 |
| torch | >=2.10 | 深度学习（D3QN） |
| pyyaml | >=6.0 | 配置文件解析 |
| Pillow | >=12.0 | 图像处理 |

### 其他技术

- **CesiumJS** - 3D可视化（导出CZML格式轨迹）
- **LaTeX (ctex)** - 中文博士论文撰写
- **Git** - 版本控制

---

## 项目结构

```
Raumfahrt/
├── Raumfahrt_Project/          # 主项目代码
│   ├── src/                    # 源代码
│   │   ├── config/             # 全局配置（物理常数、仿真参数）
│   │   ├── core/               # 核心工具（实验管理、Cesium导出、可视化）
│   │   ├── environment/        # 第三章：环境建模
│   │   │   ├── modeling.py     # EnvironmentModeling类
│   │   │   ├── terramechanics.py  # 地面力学（Bekker/Wong公式）
│   │   │   └── soil_db.py      # 土壤参数数据库
│   │   ├── dynamics/           # 第四章：动力学
│   │   │   ├── rover_dynamics.py  # LunarRoverDynamics类
│   │   │   └── estimator.py    # 参数辨识（RLS/EKF）
│   │   ├── perception/         # 感知系统（图像分割）
│   │   ├── planning/           # 第五章：路径规划
│   │   │   ├── global_planner/ # A*算法
│   │   │   │   └── astar.py
│   │   │   └── local_planner/  # D3QN强化学习
│   │   │       ├── agent.py    # D3QNAgent类
│   │   │       ├── network.py  # DuelingDQN网络结构
│   │   │       └── replay_buffer.py  # 优先经验回放
│   │   └── runtime/            # 端到端运行时
│   │       └── end_to_end.py   # 完整流程整合
│   ├── scripts/                # 执行脚本
│   │   ├── experiments/        # 各章实验脚本
│   │   │   ├── ch3_build_environment.py
│   │   │   ├── ch4_run_dynamics.py
│   │   │   ├── ch5_run_planning.py
│   │   │   └── perception_extract_features.py
│   │   ├── run_end_to_end.py   # 端到端运行入口
│   │   ├── export_cesium_path.py  # Cesium导出工具
│   │   └── train_agent.py      # RL训练脚本
│   ├── tests/                  # 单元测试
│   ├── configs/                # YAML配置文件
│   │   ├── ch3_environment.yaml
│   │   ├── ch4_dynamics.yaml
│   │   ├── ch5_planning.yaml
│   │   └── end_to_end.yaml
│   ├── apps/cesium_viewer/     # Cesium可视化前端
│   ├── outputs/                # 输出目录（git ignored）
│   └── requirements.txt        # Python依赖
├── docs/thesis/Thesis_Project/ # LaTeX论文源文件
├── push_to_texpage.py          # 论文打包上传脚本
└── test_api.py / test_codex.py # API测试脚本
```

---

## 构建与运行

### 环境配置

```bash
cd Raumfahrt_Project
pip install -r requirements.txt
```

### 运行单个章节实验

```bash
# 第三章：构建环境地图
python scripts/experiments/ch3_build_environment.py --config configs/ch3_environment.yaml

# 第四章：动力学仿真
python scripts/experiments/ch4_run_dynamics.py --config configs/ch4_dynamics.yaml

# 第五章：路径规划
python scripts/experiments/ch5_run_planning.py --config configs/ch5_planning.yaml
```

### 运行端到端流程

```bash
python scripts/run_end_to_end.py --config configs/end_to_end.yaml
```

输出目录结构：`outputs/runs/{experiment_name}/{timestamp}/`

### Cesium可视化

```bash
# 导出CZML文件
python scripts/export_cesium_path.py \
  --input outputs/runs/end_to_end/<run_id>/dynamics/dynamics_results.npz \
  --output-dir outputs/visualizations/cesium

# 启动本地服务器
cd apps/cesium_viewer
python -m http.server 8000

# 浏览器访问
# http://localhost:8000/?czml=../../outputs/visualizations/cesium/path.czml
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_end_to_end_pipeline.py
python -m pytest tests/test_ch3_environment_pipeline.py
```

---

## 代码风格规范

### 语言约定

- **主要语言**：中文（注释和文档字符串使用中文）
- **代码风格**：遵循PEP 8，但变量命名可使用中文拼音或英文
- **文档字符串**：使用Google风格

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `LunarRoverDynamics`, `D3QNAgent` |
| 函数/方法 | snake_case | `calculate_wheel_forces()` |
| 变量 | snake_case | `elevation_map`, `slip_ratio` |
| 常量 | UPPER_CASE | `LUNAR_GRAVITY`, `TIME_STEP` |
| 模块 | snake_case | `rover_dynamics.py` |

### 代码组织原则

1. **分层架构**：严格区分环境 -> 动力学 -> 规划三层
2. **配置驱动**：所有可调参数通过YAML配置文件管理
3. **管道模式**：每个模块提供`pipeline.py`封装标准流程
4. **实验可复现**：使用`src/core/experiment.py`管理随机种子、运行目录

### 文件头模板

```python
#!/usr/bin/env python3
"""
模块功能简述

详细说明（可选）
"""

# 导入按顺序：标准库 -> 第三方 -> 本地
import numpy as np
from typing import Tuple, Dict

from src.config.config import PHYSICAL_CONSTANTS
```

---

## 核心模块说明

### 1. 环境建模 (`src/environment/`)

- **modeling.py**: `EnvironmentModeling`类，管理高程图、障碍物图、可通行性图
- **terramechanics.py**: `Terramechanics`类，实现Bekker公式和Wong-Reece模型
- **soil_db.py**: 土壤物理参数数据库（月壤、压实月壤、岩石）

### 2. 动力学仿真 (`src/dynamics/`)

- **rover_dynamics.py**: `LunarRoverDynamics`类，六轮月球车完整动力学模型
  - 状态：位置、速度、姿态、角速度、轮速
  - 关键方法：`step()`, `calculate_wheel_soil_interaction()`
- **estimator.py**: 参数辨识器（RLS递归最小二乘、EKF扩展卡尔曼滤波）

### 3. 路径规划 (`src/planning/`)

- **global_planner/astar.py**: A*算法实现
  - 8方向搜索（含对角线）
  - 支持路径平滑（梯度下降法）
  - 考虑能耗的最优路径
- **local_planner/agent.py**: D3QN智能体
  - 结合A*全局路径信息作为额外特征输入
  - 优先经验回放（PER）
  - ε-greedy探索策略

### 4. 核心工具 (`src/core/`)

- **experiment.py**: 实验管理（配置加载、随机种子、运行目录创建）
- **cesium_export.py**: CZML格式导出，支持Cesium 3D可视化
- **visualization.py**: 统一绘图工具

---

## 配置系统

### 配置文件结构 (YAML)

```yaml
seed: 20260130                    # 随机种子
experiment_name: end_to_end       # 实验名称
output_root: outputs/runs         # 输出根目录
run_id: auto                      # 运行ID（auto自动生成时间戳）

environment:
  map_resolution: 1.0             # 地图分辨率 (m/pixel)
  map_size: [1000.0, 1000.0]      # 地图大小 (m)
  use_random_semantics: true      # 使用随机语义分割

planning:
  obstacle_threshold: 0.5
  start: [50.0, 50.0]
  goal: [950.0, 950.0]

dynamics:
  start_pos: [50.0, 50.0]
  end_pos: [950.0, 950.0]
  duration: 100.0
  fps: 20
  max_velocity: 1.0
  use_planning_path: true

cesium:
  enabled: true
  output_dir: outputs/visualizations/cesium
  origin: [45.0, 0.0, 0.0]        # 地理原点 [lat, lon, alt]
  sample_step: 10
  time_step: 1.0
```

### 全局常量 (`src/config/config.py`)

```python
PHYSICAL_CONSTANTS = {
    'LUNAR_GRAVITY': 1.62,        # 月球重力加速度 (m/s²)
    'EARTH_GRAVITY': 9.81,
    'AIR_DENSITY': 0.0,           # 月球大气密度 (kg/m³)
}

ROVER_PARAMS = {
    'MASS': 140.0,                # 质量 (kg)
    'WHEEL_RADIUS': 0.25,         # 车轮半径 (m)
    'WHEEL_BASE': 1.5,            # 轴距 (m)
    'MAX_LINEAR_VELOCITY': 0.5,   # 最大线速度 (m/s)
}
```

---

## 测试策略

### 测试结构

- **单元测试**：`tests/test_*.py`，使用pytest框架
- **TDD模式**：测试文件描述预期行为，先于实现编写

### 测试示例

```python
# tests/test_end_to_end_pipeline.py
def test_run_end_to_end_creates_summary(tmp_path):
    runtime = importlib.import_module("src.runtime.end_to_end")
    run_end_to_end = getattr(runtime, "run_end_to_end")
    
    config = {
        "seed": 5,
        "output_root": str(tmp_path),
        "experiment_name": "end_to_end",
        "run_id": "unit-test",
        # ...
    }
    
    result = run_end_to_end(config)
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
```

---

## 论文相关

### LaTeX论文结构

论文源文件位于 `docs/thesis/Thesis_Project/`，使用ctex宏包：

```
docs/thesis/Thesis_Project/
├── main.tex                      # 主文件
├── setup/
│   ├── preamble.tex              # 宏包加载
│   └── macros.tex                # 自定义命令
└── chapters/
    ├── 5.基于数字孪生的月面岩石识别与避障/
    │   ├── 5.1_月面岩石识别需求与挑战.tex
    │   ├── 5.2_数字孪生平台下的岩石识别框架.tex
    │   └── ...
    └── 6.基于数字孪生的巡视器路径规划/
        ├── 6.1_路径规划问题描述.tex
        └── ...
```

### 论文打包上传

```bash
python push_to_texpage.py
```

生成 `texpage_upload_package/` 目录，包含：
- `main.tex` - 合并后的主文件（第五章、第六章）
- `figures/ch5/` 和 `figures/ch6/` - 图片

---

## 安全与敏感信息

### API密钥管理

项目包含测试脚本使用OpenAI API（`test_api.py`, `test_codex.py`）：

- **注意**：脚本中硬编码了API密钥（仅用于测试）
- **生产环境**：应使用环境变量或配置文件

```python
# 推荐方式
import os
api_key = os.environ.get("OPENAI_API_KEY")
```

### Git忽略规则

```
data/           # 原始数据集（不上传）
outputs/        # 运行输出（不上传）
*.npz           # NumPy数据文件
*.pth, *.pt     # PyTorch模型权重
*.log           # 日志文件
.env            # 环境变量
```

---

## 常见问题

### Q: 如何添加新的实验配置？

1. 复制现有YAML配置文件
2. 修改参数值
3. 使用 `--config` 参数指定新配置

### Q: 如何调试动力学仿真？

动力学模块支持详细的`wheel_info`输出，包含每个车轮的：
- slip_ratio（滑移率）
- sinkage（沉陷量）
- traction（牵引力）
- rolling_resistance（滚动阻力）

### Q: 如何扩展D3QN动作空间？

修改 `src/planning/local_planner/agent.py`：
1. 修改 `self.num_actions` 数值
2. 更新 `network.py` 中的 `action_space_size`
3. 调整 `select_action()` 中的动作映射逻辑

---

## 参考资料

- `Raumfahrt_Project/README.md` - 项目详细说明
- `Raumfahrt_Project/CESIUM_VISUALIZATION.md` - Cesium可视化指南
- 论文章节Tex文件 - 算法原理说明

---

*最后更新：2026-03-02*

# Raumfahrt 项目环境配置指南

## Python版本要求

- Python >= 3.10 (推荐 3.12)
- pip >= 23.0

## 快速开始

### 1. 安装依赖

```bash
cd Raumfahrt_Project
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python scripts/env_manager.py check
```

## 手动安装

如果自动安装失败，可以手动安装核心包：

```bash
# 升级pip
python -m pip install --upgrade pip setuptools wheel

# 科学计算
pip install numpy scipy matplotlib

# 图像处理
pip install Pillow

# 配置解析
pip install pyyaml
```

## 已安装的包

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | 2.4.2 | 数值计算 |
| scipy | 1.17.1 | 科学计算（高斯滤波等）|
| matplotlib | 3.10.8 | 数据可视化 |
| Pillow | 12.1.1 | 图像处理 |
| PyYAML | 6.0.3 | YAML配置解析 |

## 环境管理脚本

### 检查环境
```bash
python scripts/env_manager.py check
```

### 安装所有依赖
```bash
python scripts/env_manager.py install
```

### 安装单个包
```bash
python scripts/env_manager.py install-scipy
```

## 常见问题

### Q: pip安装速度慢
**A:** 使用国内镜像源
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 安装scipy失败
**A:** 确保已安装Microsoft Visual C++ Redistributable，或使用conda
```bash
conda install scipy
```

### Q: 中文字体显示问题
**A:** 确保系统安装Microsoft YaHei或SimHei字体

## Windows PowerShell 配置

如果pip命令不可用，添加Python Scripts到PATH：

```powershell
# 临时添加
$env:PATH += ";C:\Users\Don_Wang\AppData\Roaming\Python\Python312\Scripts"

# 永久添加（PowerShell 7+）
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Users\Don_Wang\AppData\Roaming\Python\Python312\Scripts", "User")
```

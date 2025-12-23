# -*- coding: utf-8 -*-
"""
工具函数模块 - 通用工具函数集合
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import json


def set_random_seed(seed: int):
    """
    设置全局随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"随机种子设置为: {seed}")


def create_directories(*paths: str):
    """
    创建多个目录

    Args:
        *paths: 目录路径列表
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


class Logger:
    """
    日志记录器
    同时输出到控制台和文件
    """

    def __init__(self, log_file: str, print_to_console: bool = True):
        """
        初始化日志记录器

        Args:
            log_file: 日志文件路径
            print_to_console: 是否同时输出到控制台
        """
        self.log_file = log_file
        self.print_to_console = print_to_console

        # 创建日志目录
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 清空或创建日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"日志开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        """
        记录日志消息

        Args:
            message: 日志消息
            level: 日志级别
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] [{level}] {message}"

        # 输出到控制台
        if self.print_to_console:
            print(message)

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')

    def info(self, message: str):
        """记录INFO级别日志"""
        self.log(message, "INFO")

    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.log(message, "WARNING")

    def error(self, message: str):
        """记录ERROR级别日志"""
        self.log(message, "ERROR")

    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.log(message, "DEBUG")


class MovingAverage:
    """
    移动平均计算器
    用于平滑训练曲线
    """

    def __init__(self, window_size: int = 100):
        """
        初始化移动平均计算器

        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        self.values = []

    def add(self, value: float):
        """添加新值"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get(self) -> float:
        """获取当前移动平均值"""
        if not self.values:
            return 0.0
        return np.mean(self.values)

    def reset(self):
        """重置"""
        self.values = []


class MetricsTracker:
    """
    训练指标追踪器
    """

    def __init__(self):
        """初始化指标追踪器"""
        self.metrics = {}
        self.history = {}

    def add(self, name: str, value: float, step: int = None):
        """
        添加指标值

        Args:
            name: 指标名称
            value: 指标值
            step: 步数（可选）
        """
        if name not in self.metrics:
            self.metrics[name] = MovingAverage()
            self.history[name] = []

        self.metrics[name].add(value)
        self.history[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })

    def get_average(self, name: str) -> float:
        """获取指标的移动平均值"""
        if name in self.metrics:
            return self.metrics[name].get()
        return 0.0

    def get_latest(self, name: str) -> float:
        """获取指标的最新值"""
        if name in self.history and self.history[name]:
            return self.history[name][-1]['value']
        return 0.0

    def get_history(self, name: str) -> List[float]:
        """获取指标的历史值列表"""
        if name in self.history:
            return [h['value'] for h in self.history[name]]
        return []

    def save(self, filepath: str):
        """保存指标历史到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """从文件加载指标历史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-π, π] 范围

    Args:
        angle: 输入角度（弧度）

    Returns:
        归一化后的角度
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    计算两点之间的欧几里得距离

    Args:
        p1: 点1坐标 (x, y)
        p2: 点2坐标 (x, y)

    Returns:
        距离
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    计算两点之间的曼哈顿距离

    Args:
        p1: 点1坐标
        p2: 点2坐标

    Returns:
        曼哈顿距离
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def interpolate_path(path: List[Tuple[float, float]],
                     num_points: int) -> List[Tuple[float, float]]:
    """
    对路径进行插值，生成更平滑的路径

    Args:
        path: 原始路径点列表
        num_points: 目标点数

    Returns:
        插值后的路径
    """
    if len(path) < 2:
        return path

    # 计算原始路径总长度
    path_lengths = [0.0]
    for i in range(1, len(path)):
        dist = euclidean_distance(path[i - 1], path[i])
        path_lengths.append(path_lengths[-1] + dist)

    total_length = path_lengths[-1]

    # 生成均匀分布的采样点
    sample_distances = np.linspace(0, total_length, num_points)
    interpolated_path = []

    for d in sample_distances:
        # 找到d所在的路径段
        for i in range(len(path_lengths) - 1):
            if path_lengths[i] <= d <= path_lengths[i + 1]:
                # 线性插值
                segment_length = path_lengths[i + 1] - path_lengths[i]
                if segment_length > 0:
                    t = (d - path_lengths[i]) / segment_length
                    x = path[i][0] + t * (path[i + 1][0] - path[i][0])
                    y = path[i][1] + t * (path[i + 1][1] - path[i][1])
                    interpolated_path.append((x, y))
                else:
                    interpolated_path.append(path[i])
                break

    return interpolated_path


def calculate_path_length(path: List[Tuple[float, float]]) -> float:
    """
    计算路径总长度

    Args:
        path: 路径点列表

    Returns:
        路径长度
    """
    if len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(1, len(path)):
        total_length += euclidean_distance(path[i - 1], path[i])

    return total_length


def calculate_path_smoothness(path: List[Tuple[float, float]]) -> float:
    """
    计算路径平滑度（转角变化的累积）

    Args:
        path: 路径点列表

    Returns:
        平滑度指标（值越小越平滑）
    """
    if len(path) < 3:
        return 0.0

    total_angle_change = 0.0
    for i in range(1, len(path) - 1):
        # 计算两个相邻路径段的方向
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]
        dx2 = path[i + 1][0] - path[i][0]
        dy2 = path[i + 1][1] - path[i][1]

        # 计算角度变化
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        angle_diff = abs(normalize_angle(angle2 - angle1))

        total_angle_change += angle_diff

    return total_angle_change


def soft_update_params(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    """
    软更新目标网络参数

    θ_target = τ * θ_source + (1 - τ) * θ_target

    Args:
        source: 源网络
        target: 目标网络
        tau: 软更新系数 (0 < tau <= 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update_params(source: torch.nn.Module, target: torch.nn.Module):
    """
    硬更新目标网络参数（完全复制）

    Args:
        source: 源网络
        target: 目标网络
    """
    target.load_state_dict(source.state_dict())


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_learning_curve(rewards: List[float], save_path: str,
                        window: int = 50, title: str = "Learning Curve"):
    """
    绘制学习曲线

    Args:
        rewards: 回合奖励列表
        save_path: 保存路径
        window: 移动平均窗口大小
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 原始数据
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')

    # 移动平均
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(rewards)), moving_avg,
                color='red', linewidth=2, label=f'{window}-Episode Moving Average')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def format_time(seconds: float) -> str:
    """
    将秒数格式化为可读字符串

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_model_summary(model: torch.nn.Module, input_shapes: Dict[str, Tuple]):
    """
    打印模型摘要

    Args:
        model: PyTorch模型
        input_shapes: 输入形状字典
    """
    print("\n" + "=" * 60)
    print("模型摘要")
    print("=" * 60)

    print(f"\n模型结构:")
    print(model)

    total_params = count_parameters(model)
    print(f"\n可训练参数数量: {total_params:,}")

    print("\n输入形状:")
    for name, shape in input_shapes.items():
        print(f"  {name}: {shape}")

    print("=" * 60 + "\n")


class EarlyStopping:
    """
    早停机制
    当验证指标不再改善时停止训练
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'max'):
        """
        初始化早停机制

        Args:
            patience: 容忍的无改善轮数
            min_delta: 最小改善阈值
            mode: 'max'表示指标越大越好，'min'表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """
        检查是否应该早停

        Args:
            value: 当前指标值

        Returns:
            是否应该早停
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False


# 测试代码
if __name__ == "__main__":
    print("=== 工具函数测试 ===\n")

    # 测试随机种子设置
    set_random_seed(42)

    # 测试日志记录器
    logger = Logger('./test_logs/test.log')
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")

    # 测试移动平均
    ma = MovingAverage(window_size=10)
    for i in range(20):
        ma.add(i)
        print(f"值: {i}, 移动平均: {ma.get():.2f}")

    # 测试指标追踪器
    tracker = MetricsTracker()
    for i in range(100):
        tracker.add('reward', np.random.randn() + i * 0.1, step=i)
    print(f"\n奖励平均值: {tracker.get_average('reward'):.2f}")

    # 测试路径计算
    path = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
    print(f"\n路径长度: {calculate_path_length(path):.2f}")
    print(f"路径平滑度: {calculate_path_smoothness(path):.4f}")

    # 测试时间格式化
    print(f"\n时间格式化: {format_time(3723.5)}")

    # 测试早停
    early_stop = EarlyStopping(patience=3)
    values = [1, 2, 3, 2.9, 2.8, 2.7, 2.6]
    for v in values:
        stop = early_stop(v)
        print(f"值: {v}, 应该早停: {stop}")

    print("\n工具函数测试完成!")

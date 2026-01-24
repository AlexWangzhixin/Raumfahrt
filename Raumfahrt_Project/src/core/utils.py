#!/usr/bin/env python3
"""
通用工具函数模块
提供各种辅助函数
"""

import numpy as np
import os
import json
import yaml
from datetime import datetime

# 文件操作函数
def ensure_directory(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)

def load_json(file_path):
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path, indent=2):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_yaml(file_path):
    """
    加载YAML文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path):
    """
    保存数据到YAML文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

# 数学工具函数
def normalize_angle(angle):
    """
    归一化角度到[-π, π]
    
    Args:
        angle: 角度（弧度）
    
    Returns:
        归一化后的角度
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def calculate_distance(point1, point2):
    """
    计算两点之间的欧几里得距离
    
    Args:
        point1: 第一个点 (x1, y1)
        point2: 第二个点 (x2, y2)
    
    Returns:
        距离
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_heading(from_point, to_point):
    """
    计算从from_point到to_point的航向角
    
    Args:
        from_point: 起点 (x1, y1)
        to_point: 终点 (x2, y2)
    
    Returns:
        航向角（弧度）
    """
    return np.arctan2(to_point[1] - from_point[1], to_point[0] - from_point[0])

# 坐标转换函数
def cartesian_to_polar(x, y):
    """
    笛卡尔坐标转极坐标
    
    Args:
        x: x坐标
        y: y坐标
    
    Returns:
        (r, theta): 极径和极角（弧度）
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    """
    极坐标转笛卡尔坐标
    
    Args:
        r: 极径
        theta: 极角（弧度）
    
    Returns:
        (x, y): 笛卡尔坐标
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# 时间工具函数
def get_timestamp():
    """
    获取当前时间戳
    
    Returns:
        时间戳字符串
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_duration(seconds):
    """
    格式化时间 duration
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

# 数据处理函数
def smooth_trajectory(trajectory, window_size=5):
    """
    平滑轨迹
    
    Args:
        trajectory: 轨迹点列表 [(x1, y1), (x2, y2), ...]
        window_size: 窗口大小
    
    Returns:
        平滑后的轨迹
    """
    if len(trajectory) < window_size:
        return trajectory
    
    trajectory = np.array(trajectory)
    smoothed = np.zeros_like(trajectory)
    
    for i in range(len(trajectory)):
        start = max(0, i - window_size // 2)
        end = min(len(trajectory), i + window_size // 2 + 1)
        smoothed[i] = np.mean(trajectory[start:end], axis=0)
    
    return smoothed.tolist()

def downsample_trajectory(trajectory, factor=2):
    """
    下采样轨迹
    
    Args:
        trajectory: 轨迹点列表 [(x1, y1), (x2, y2), ...]
        factor: 下采样因子
    
    Returns:
        下采样后的轨迹
    """
    return trajectory[::factor]

# 导出所有函数
__all__ = [
    'ensure_directory',
    'load_json',
    'save_json',
    'load_yaml',
    'save_yaml',
    'normalize_angle',
    'calculate_distance',
    'calculate_heading',
    'cartesian_to_polar',
    'polar_to_cartesian',
    'get_timestamp',
    'format_duration',
    'smooth_trajectory',
    'downsample_trajectory',
]

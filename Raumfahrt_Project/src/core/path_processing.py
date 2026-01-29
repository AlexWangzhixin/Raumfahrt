#!/usr/bin/env python3
"""
路径处理模块

功能：
1. B-spline路径平滑插值
2. 路径数据加载与处理
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev


def smooth_path_bspline(x, y, num_points=500, smoothing=0.1):
    """
    使用 B样条 (B-Spline) 对稀疏路点进行平滑插值
    
    Args:
        x, y: 原始路点的坐标列表
        num_points: 插值后的点数 (越多越平滑)
        smoothing: 平滑因子 (0表示强制经过所有点，越大越平滑但可能偏离原点)
        
    Returns:
        np.array: 平滑后的 [[x0,y0], [x1,y1], ...]
    """
    # 移除重复点，否则 splprep 会报错
    points = np.vstack((x, y)).T
    _, idx = np.unique(points, axis=0, return_index=True)
    # 保持原有顺序
    idx = np.sort(idx)
    x = points[idx, 0]
    y = points[idx, 1]

    # 如果点太少，无法进行高阶插值，直接返回线性插值
    if len(x) < 3:
        return np.column_stack((
            np.linspace(x[0], x[-1], num_points),
            np.linspace(y[0], y[-1], num_points)
        ))

    # B样条插值
    # k=3 表示三次样条 (Cubic Spline)，曲线最自然
    try:
        tck, u = splprep([x, y], s=smoothing, k=3)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        result = np.column_stack((x_new, y_new))
        # 确保起点和终点与原始点一致
        result[0] = [x[0], y[0]]
        result[-1] = [x[-1], y[-1]]
        return result
    except Exception as e:
        print(f"插值失败 (可能点太少或共线): {e}, 退化为线性插值")
        # 即使插值失败，也按照指定的num_points返回结果
        return np.column_stack((
            np.linspace(x[0], x[-1], num_points),
            np.linspace(y[0], y[-1], num_points)
        ))


def load_real_path_data(file_path):
    """
    加载并平滑路径数据
    """
    print(f"正在加载真实路径数据: {file_path}")
    
    raw_points = []
    
    if not os.path.exists(file_path):
        print(f"警告: 路径文件不存在，使用【模拟真实规划路点】...")
        # 模拟真实的"规划关键点" (Key Waypoints)
        # 这些点代表地面站规划的转弯点
        raw_points = [
            [0.0, 0.0],    # 着陆点
            [5.0, 2.0],    # 避障点A
            [12.0, 1.5],   # 科学探测点B
            [18.0, 4.0],   # 陨石坑边缘C
            [22.0, 5.5],   # 调整航向
            [28.0, 3.0],   # 目的地D
            [32.0, 0.0]    # 最终休眠点
        ]
        raw_data = np.array(raw_points)
    else:
        try:
            df = pd.read_csv(file_path)
            raw_data = df[['x', 'y']].values
        except Exception as e:
            print(f"读取失败: {e}")
            sys.exit(1)

    # === 关键修正：进行平滑插值 ===
    print("正在对离散路点进行 B-Spline 平滑处理...")
    smooth_path = smooth_path_bspline(
        raw_data[:, 0],
        raw_data[:, 1],
        num_points=1000, # 生成1000个细密点
        smoothing=0.5    # 允许轻微偏离原点以换取曲线平滑度
    )
    
    return smooth_path

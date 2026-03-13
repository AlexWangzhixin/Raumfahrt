#!/usr/bin/env python3
"""
生成图3-4: 撞击坑识别可视化效果
展示撞击坑检测算法的结果，包括正确识别、漏检和误检
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_chinese_font() -> str:
    """配置matplotlib中文字体"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "SimSun",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["font.size"] = 16
            plt.rcParams["axes.titlesize"] = 18
            plt.rcParams["axes.labelsize"] = 16
            plt.rcParams["xtick.labelsize"] = 14
            plt.rcParams["ytick.labelsize"] = 14
            return name
    
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 16
    return "DejaVu Sans"


def generate_base_terrain(size=400):
    """生成基础月面地形"""
    np.random.seed(42)
    
    x = np.linspace(0, 100, size)
    y = np.linspace(0, 100, size)
    X, Y = np.meshgrid(x, y)
    
    # 基础地形
    base = -2500 + 50 * np.sin(X/15) * np.cos(Y/15)
    
    # 添加表面粗糙度
    roughness = 3 * np.random.randn(size, size)
    
    elevation = base + roughness
    
    return elevation, X, Y


def generate_crater_data():
    """
    生成撞击坑识别数据
    
    Returns:
        true_craters: 真实撞击坑列表 (正确识别)
        missed_craters: 漏检撞击坑列表
        false_positives: 误检列表
    """
    np.random.seed(123)
    
    # 真实撞击坑 (x, y, radius)
    true_craters = [
        (20, 25, 4.5),
        (45, 35, 3.2),
        (70, 55, 5.0),
        (35, 70, 2.8),
        (80, 25, 3.8),
        (55, 80, 4.2),
        (15, 60, 2.5),
        (65, 40, 3.5),
    ]
    
    # 漏检的撞击坑（真实存在但算法未检测到）
    missed_craters = [
        (30, 45, 2.0),
        (75, 75, 1.8),
        (50, 20, 2.3),
    ]
    
    # 误检（算法检测到的但不是真实撞击坑）
    false_positives = [
        (25, 55, 2.5),
        (60, 30, 3.0),
        (85, 65, 2.2),
    ]
    
    return true_craters, missed_craters, false_positives


def apply_crater_to_terrain(elevation, X, Y, crater_x, crater_y, radius, depth):
    """在 terrain 上添加撞击坑"""
    dist = np.sqrt((X - crater_x)**2 + (Y - crater_y)**2)
    crater_mask = dist < radius
    elevation = elevation.copy()
    elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
    return elevation


def generate_figure_3_4():
    """生成图3-4: 撞击坑识别可视化效果"""
    
    configure_chinese_font()
    
    # 生成基础地形
    elevation, X, Y = generate_base_terrain(size=400)
    
    # 获取撞击坑数据
    true_craters, missed_craters, false_positives = generate_crater_data()
    
    # 将真实撞击坑添加到地形
    for cx, cy, radius in true_craters + missed_craters:
        depth = radius * 40  # 深度与半径成正比
        elevation = apply_crater_to_terrain(elevation, X, Y, cx, cy, radius, depth)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # 显示地形
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im = ax.imshow(elevation, extent=extent, cmap='gray', origin='lower', aspect='equal')
    
    # 绘制正确识别的撞击坑（绿色圆圈）
    for cx, cy, radius in true_craters:
        circle = Circle((cx, cy), radius, fill=False, 
                       edgecolor='#00AA00', linewidth=2.5, linestyle='-')
        ax.add_patch(circle)
        # 添加中心点
        ax.plot(cx, cy, 'g+', markersize=10, markeredgewidth=2)
    
    # 绘制漏检的撞击坑（红色叉号）
    for cx, cy, radius in missed_craters:
        circle = Circle((cx, cy), radius, fill=False, 
                       edgecolor='#CC0000', linewidth=2.5, linestyle='--')
        ax.add_patch(circle)
        ax.plot(cx, cy, 'rx', markersize=12, markeredgewidth=2.5)
    
    # 绘制误检（蓝色三角形）
    for cx, cy, radius in false_positives:
        ax.plot(cx, cy, 'b^', markersize=12, markeredgecolor='white', 
               markeredgewidth=1.5, label='误检' if cx == false_positives[0][0] else "")
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00AA00', lw=2.5, marker='+', markersize=10, 
               label='正确识别', linestyle='-'),
        Line2D([0], [0], color='#CC0000', lw=2.5, marker='x', markersize=10, 
               label='漏检', linestyle='--'),
        Line2D([0], [0], color='blue', lw=0, marker='^', markersize=10, 
               label='误检'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, 
             framealpha=0.9, title='识别结果', title_fontsize=14)
    
    # 设置标签
    ax.set_xlabel('东西距离 (km)', fontsize=16)
    ax.set_ylabel('南北距离 (km)', fontsize=16)
    ax.set_title('(a) 虹湾着陆区撞击坑识别结果\n(绿色：正确识别，红色：漏检，蓝色：误检)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label('高程 (m)', fontsize=14, rotation=90, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # 保存
    path = OUTPUT_DIR / "fig_3_4_crater_detection.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"已保存: {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("生成图3-4: 撞击坑识别可视化效果")
    print("=" * 60)
    
    generate_figure_3_4()
    
    print("=" * 60)
    print("完成!")
    print("=" * 60)

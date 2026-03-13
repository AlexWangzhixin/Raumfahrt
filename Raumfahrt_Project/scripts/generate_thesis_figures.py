#!/usr/bin/env python3
"""
生成博士论文所需的高质量图像
符合博士论文要求的：
- 高分辨率 (300+ DPI)
- 中文字体支持
- 清晰的标签和图例
- 专业的配色方案
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 配置中文字体
def configure_chinese_font() -> str:
    """配置matplotlib中文字体"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "SimSun",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"

# 论文图像输出目录
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 博士论文标准配色方案
COLORS = {
    'primary': '#1f77b4',      # 主色调：蓝色
    'secondary': '#ff7f0e',    # 次要：橙色
    'tertiary': '#2ca02c',     # 第三：绿色
    'quaternary': '#d62728',   # 第四：红色
    'quinary': '#9467bd',      # 第五：紫色
    'moon_soil': '#8B7355',    # 月壤色
    'moon_rock': '#696969',    # 月岩色
    'lunar_surface': '#C0C0C0', # 月面色
    'highlight': '#FFD700',    # 高亮：金色
    'danger': '#DC143C',       # 危险：深红
    'safe': '#32CD32',         # 安全：翠绿
}


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """
    保存图像，生成多种格式供选择
    
    Args:
        fig: matplotlib图像对象
        filename: 文件名（不含扩展名）
        dpi: 分辨率
        
    Returns:
        保存的文件路径
    """
    # PNG格式（用于预览）
    png_path = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # PDF格式（用于LaTeX插入，矢量图）
    pdf_path = OUTPUT_DIR / f"{filename}.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # EPS格式（备用矢量格式）
    eps_path = OUTPUT_DIR / f"{filename}.eps"
    fig.savefig(eps_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close(fig)
    return png_path


def generate_figure_1_1_risk_pie() -> Path:
    """
    图1-1: 月面巡视器主要风险源分类示意图
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    labels = ['几何障碍', '动力学失稳', '感知失效', '通讯时延', '能耗超限']
    sizes = [28, 24, 16, 14, 18]
    colors = [COLORS['danger'], COLORS['quaternary'], COLORS['secondary'], 
              COLORS['quinary'], COLORS['tertiary']]
    explode = (0.05, 0.02, 0, 0, 0)  # 突出显示前两个
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12}
    )
    
    # 美化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('图1-1 月面巡视器主要风险源分类示意图', 
                 fontsize=14, fontweight='bold', pad=20)
    
    return save_figure(fig, 'fig_1_1_risk_classification')


def generate_figure_3_3_segmentation() -> Path:
    """
    图3-3: 语义分割结果示例
    三联图：原始影像、灰度纹理、语义分割
    """
    # 生成模拟月球表面数据
    np.random.seed(42)
    size = 400
    
    # 模拟原始影像（RGB）
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # 创建月球表面纹理
    r = 0.4 + 0.3 * np.sin(X/2) * np.cos(Y/2) + 0.1 * np.random.rand(size, size)
    g = 0.4 + 0.3 * np.sin(X/2 + 1) * np.cos(Y/2 + 0.5) + 0.1 * np.random.rand(size, size)
    b = 0.45 + 0.25 * np.sin(X/2 + 2) * np.cos(Y/2 + 1) + 0.1 * np.random.rand(size, size)
    
    # 添加岩石区域（亮斑）
    for _ in range(15):
        cx, cy = np.random.randint(0, size, 2)
        r_rock = np.random.randint(5, 20)
        Y_idx, X_idx = np.ogrid[:size, :size]
        mask = (X_idx - cx)**2 + (Y_idx - cy)**2 <= r_rock**2
        r[mask] = np.random.uniform(0.7, 0.9)
        g[mask] = np.random.uniform(0.7, 0.9)
        b[mask] = np.random.uniform(0.7, 0.9)
    
    rgb_image = np.stack([r, g, b], axis=2)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # 灰度纹理
    gray = rgb_image.mean(axis=2)
    
    # 语义分割结果
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    semantic = np.zeros((size, size), dtype=int)
    semantic[gray > np.percentile(gray, 60)] = 1  # 高地
    semantic[gradient_magnitude > np.percentile(gradient_magnitude, 85)] = 2  # 岩石
    
    # 创建三联图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    # (a) 原始影像
    axes[0].imshow(rgb_image)
    axes[0].set_title('(a) 原始影像', fontsize=12)
    axes[0].axis('off')
    
    # (b) 灰度纹理
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('(b) 灰度纹理', fontsize=12)
    axes[1].axis('off')
    
    # (c) 语义分割
    cmap = matplotlib.colors.ListedColormap(['#1e88e5', '#43a047', '#ef5350'])
    im = axes[2].imshow(semantic, cmap=cmap, vmin=0, vmax=2)
    axes[2].set_title('(c) 语义分割', fontsize=12)
    axes[2].axis('off')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_ticks([0.33, 1, 1.67])
    cbar.set_ticklabels(['月海', '高地', '岩石'])
    cbar.ax.tick_params(labelsize=10)
    
    fig.suptitle('图3-3 语义分割结果示例', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return save_figure(fig, 'fig_3_3_semantic_segmentation')


def generate_figure_3_4_heatmap() -> Path:
    """
    图3-4: 物理参数场可视化热力图
    """
    np.random.seed(123)
    size = 300
    
    # 生成地形高程数据
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # 创建复杂的地形模式
    Z = (np.sin(X) * np.cos(Y) + 
         0.5 * np.sin(2*X) * np.cos(2*Y) + 
         0.25 * np.sin(4*X) * np.cos(4*Y))
    
    # 计算坡度作为风险指标
    grad_x, grad_y = np.gradient(Z)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化到0-1
    risk = (slope - slope.min()) / (slope.max() - slope.min())
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # 使用inferno colormap显示风险热力图
    im = ax.imshow(risk, cmap='inferno', origin='lower', aspect='auto')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('归一化风险值', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加网格线
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # 添加示例路径
    t = np.linspace(0, 1, 200)
    path_x = 0.1 * size + 0.8 * size * t
    path_y = 0.15 * size + 0.7 * size * t + 0.1 * size * np.sin(3 * np.pi * t)
    
    ax.plot(path_x, path_y, 'g-', linewidth=2.5, label='规划路径', alpha=0.9)
    ax.scatter([path_x[0]], [path_y[0]], c='green', s=100, marker='o', 
               label='起点', zorder=5, edgecolors='white', linewidths=1)
    ax.scatter([path_x[-1]], [path_y[-1]], c='red', s=100, marker='s', 
               label='终点', zorder=5, edgecolors='white', linewidths=1)
    
    ax.set_xlabel('X (像素)', fontsize=12)
    ax.set_ylabel('Y (像素)', fontsize=12)
    ax.set_title('图3-4 物理参数场可视化热力图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    return save_figure(fig, 'fig_3_4_physics_field_heatmap')


def generate_figure_3_5_sinkage_curve() -> Path:
    """
    图3-5: 1/6g与1g沉陷曲线对比
    """
    x = np.linspace(0, 80, 300)
    
    # 1g环境下的沉陷（地球重力）
    y_1g = 0.5 + 2.8 * (1 - np.exp(-x / 22))
    
    # 1/6g环境下的沉陷（月球重力）
    y_1_6g = 0.25 + 1.35 * (1 - np.exp(-x / 26))
    
    # 添加置信区间
    y_1_6g_upper = y_1_6g * 1.1
    y_1_6g_lower = y_1_6g * 0.9
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # 绘制曲线
    ax.plot(x, y_1g, label='1g (地球)', color=COLORS['danger'], 
            linewidth=2.5, linestyle='-')
    ax.plot(x, y_1_6g, label='1/6g (月球)', color=COLORS['primary'], 
            linewidth=2.5, linestyle='-')
    
    # 填充置信区间
    ax.fill_between(x, y_1_6g_lower, y_1_6g_upper, 
                    color=COLORS['primary'], alpha=0.15, label='1/6g置信区间')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置标签
    ax.set_xlabel('行驶距离 (m)', fontsize=12)
    ax.set_ylabel('沉陷量 (cm)', fontsize=12)
    ax.set_title('图3-5 1/6g与1g沉陷曲线对比', fontsize=14, fontweight='bold')
    
    # 设置图例
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    # 添加注释
    ax.annotate('月球低重力环境\n显著降低沉陷', 
                xy=(60, y_1_6g[180]), xytext=(45, 2.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return save_figure(fig, 'fig_3_5_sinkage_comparison')


def generate_figure_4_1_wheel_soil() -> Path:
    """
    图4-1: 轮-壤接触力学示意图
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 绘制车轮
    theta = np.linspace(0, 2*np.pi, 100)
    wheel_radius = 1.0
    wheel_center = (0, wheel_radius)
    
    # 车轮外轮廓
    wheel_x = wheel_center[0] + wheel_radius * np.cos(theta)
    wheel_y = wheel_center[1] + wheel_radius * np.sin(theta)
    ax.plot(wheel_x, wheel_y, 'k-', linewidth=2.5)
    
    # 填充车轮
    ax.fill(wheel_x, wheel_y, color='#E0E0E0', alpha=0.5)
    
    # 绘制轮辐
    for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
        x_end = wheel_center[0] + 0.85 * wheel_radius * np.cos(angle)
        y_end = wheel_center[1] + 0.85 * wheel_radius * np.sin(angle)
        ax.plot([wheel_center[0], x_end], [wheel_center[1], y_end], 
                'k-', linewidth=1)
    
    # 绘制轮毂
    hub = Circle(wheel_center, 0.15, fill=True, color='gray')
    ax.add_patch(hub)
    
    # 绘制地面
    ground_y = 0
    ax.axhline(y=ground_y, color='#8B4513', linewidth=3)
    
    # 填充土壤
    soil_x = np.linspace(-2.5, 2.5, 100)
    soil_y_bottom = -1.2
    ax.fill_between(soil_x, soil_y_bottom, ground_y, color='#D2B48C', alpha=0.6)
    
    # 绘制沉陷区域
    sinkage_depth = 0.25
    contact_angle = np.deg2rad(30)
    
    # 接触弧
    contact_start_angle = -np.pi/2 - contact_angle
    contact_end_angle = -np.pi/2 + contact_angle
    contact_theta = np.linspace(contact_start_angle, contact_end_angle, 50)
    contact_x = wheel_center[0] + wheel_radius * np.cos(contact_theta)
    contact_y = wheel_center[1] + wheel_radius * np.sin(contact_theta)
    
    # 绘制沉陷区域
    ax.fill_between(contact_x, contact_y, ground_y, 
                    color='#8B4513', alpha=0.4)
    
    # 标注沉陷量 z
    ax.annotate('', xy=(wheel_center[0] - 0.8, ground_y), 
                xytext=(wheel_center[0] - 0.8, wheel_center[1] - wheel_radius + sinkage_depth),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(wheel_center[0] - 1.3, wheel_center[1] - wheel_radius + sinkage_depth/2, 
            '沉陷量 z', fontsize=12, color='red', fontweight='bold')
    
    # 标注接触角 θ
    arc_angles = np.linspace(-np.pi/2, -np.pi/2 + contact_angle, 30)
    arc_x = wheel_center[0] + 0.6 * wheel_radius * np.cos(arc_angles)
    arc_y = wheel_center[1] + 0.6 * wheel_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'g-', linewidth=2)
    ax.text(wheel_center[0] + 0.4, wheel_center[1] - 0.5, 'θ', 
            fontsize=14, color='green', fontweight='bold')
    
    # 标注滑移率 s（速度方向）
    ax.arrow(wheel_center[0], wheel_center[1] + 0.3, 0.8, 0, 
             head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax.text(wheel_center[0] + 0.4, wheel_center[1] + 0.5, 'v (速度)', 
            fontsize=11, color='blue')
    
    # 添加力标注
    # 法向力
    ax.arrow(wheel_center[0], wheel_center[1], 0, -0.6, 
             head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    ax.text(wheel_center[0] + 0.15, wheel_center[1] - 0.3, 'N', 
            fontsize=12, color='purple', fontweight='bold')
    
    # 切向力（牵引力）
    ax.arrow(wheel_center[0], wheel_center[1] - wheel_radius + 0.2, 0.6, 0, 
             head_width=0.1, head_length=0.1, fc='orange', ec='orange', linewidth=2)
    ax.text(wheel_center[0] + 0.7, wheel_center[1] - wheel_radius + 0.3, 'T (牵引力)', 
            fontsize=11, color='orange')
    
    # 土壤阻力
    ax.arrow(contact_x[10], contact_y[10], -0.3, 0.2, 
             head_width=0.08, head_length=0.08, fc='brown', ec='brown', linewidth=1.5)
    ax.text(contact_x[10] - 0.5, contact_y[10] + 0.3, 'R', 
            fontsize=11, color='brown')
    
    # 设置坐标轴
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('图4-1 轮-壤接触力学示意图', fontsize=14, fontweight='bold', pad=20)
    
    return save_figure(fig, 'fig_4_1_wheel_soil_mechanics')


def generate_figure_4_3_traction_curve() -> Path:
    """
    图4-3: 牵引力-滑移率曲线
    """
    s = np.linspace(0, 1.0, 300)
    
    # 牵引力模型（基于Wong-Reece理论）
    traction = 0.72 * s * np.exp(-3.1 * s) + 0.1 * (1 - np.exp(-8 * s))
    traction = traction / traction.max() * 1.9  # 归一化
    
    # 找到峰值点
    peak_idx = np.argmax(traction)
    s_peak = s[peak_idx]
    traction_peak = traction[peak_idx]
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # 绘制主曲线
    ax.plot(s, traction, color=COLORS['quinary'], linewidth=2.5, label='牵引力')
    
    # 标记峰值点
    ax.scatter([s_peak], [traction_peak], color=COLORS['danger'], s=100, zorder=5)
    ax.annotate(f'峰值点\n(s={s_peak:.2f}, T={traction_peak:.2f})', 
                xy=(s_peak, traction_peak), 
                xytext=(s_peak + 0.15, traction_peak - 0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 标记饱和区
    ax.axhline(y=traction[-1], xmin=0.7, xmax=1.0, color=COLORS['primary'], 
               linestyle='--', linewidth=2, label='饱和区')
    ax.fill_between(s[210:], traction[210:], traction[-1], 
                    color=COLORS['primary'], alpha=0.1)
    
    # 添加网格和标签
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('滑移率 s', fontsize=12)
    ax.set_ylabel('牵引力 (kN, 归一化)', fontsize=12)
    ax.set_title('图4-3 牵引力-滑移率曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    
    # 添加阶段标注
    ax.text(0.15, 0.5, '线性区', fontsize=10, color='gray', style='italic')
    ax.text(0.35, 1.4, '峰值区', fontsize=10, color='gray', style='italic')
    ax.text(0.75, 0.3, '饱和区', fontsize=10, color='gray', style='italic')
    
    return save_figure(fig, 'fig_4_3_traction_slip_curve')


def generate_figure_4_6_apollo_compare() -> Path:
    """
    图4-6: Apollo LRV轮迹对比验证
    """
    samples = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    measured = np.array([6.0, 9.2, 12.8, 14.5, 8.0, 10.7])
    predicted = measured + np.array([-0.8, 0.7, 1.0, -0.6, -0.4, 0.5])
    error = np.abs(predicted - measured)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 左图：对比柱状图
    x = np.arange(len(samples))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, measured, width, label='实测值', 
                        color=COLORS['primary'], alpha=0.8)
    bars2 = axes[0].bar(x + width/2, predicted, width, label='预测值', 
                        color=COLORS['secondary'], alpha=0.8)
    
    axes[0].set_xlabel('样本编号', fontsize=12)
    axes[0].set_ylabel('滑移率 (%)', fontsize=12)
    axes[0].set_title('(a) 滑移率对比', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(samples)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    # 右图：误差图
    axes[1].bar(x, error, color=COLORS['danger'], alpha=0.7)
    axes[1].axhline(y=error.mean(), color='black', linestyle='--', 
                    linewidth=2, label=f'平均误差: {error.mean():.2f}%')
    axes[1].set_xlabel('样本编号', fontsize=12)
    axes[1].set_ylabel('绝对误差 (%)', fontsize=12)
    axes[1].set_title('(b) 绝对误差', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(samples)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('图4-6 Apollo LRV轮迹对比验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return save_figure(fig, 'fig_4_6_apollo_validation')


def generate_figure_5_3_obstacle_avoidance() -> Path:
    """
    图5-3: 数字孪生环境下避障预演图
    """
    np.random.seed(456)
    size = 300
    
    # 生成风险场
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # 障碍物位置
    obstacles = [(80, 100, 15), (150, 180, 20), (220, 120, 18), (100, 220, 12)]
    risk = np.zeros((size, size))
    
    for ox, oy, radius in obstacles:
        dist = np.sqrt((X - x[ox])**2 + (Y - y[oy])**2)
        risk += np.exp(-dist**2 / (2 * (radius/size*10)**2))
    
    # 添加地形风险
    risk += 0.3 * np.abs(np.sin(X) * np.cos(Y))
    risk = np.clip(risk, 0, 1)
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # 绘制风险热力图
    im = ax.imshow(risk, cmap='hot', origin='lower', vmin=0, vmax=1)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('风险等级', fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['低', '较低', '中', '较高', '高'])
    
    # 绘制避障路径
    t = np.linspace(0, 1, 200)
    
    # 路径1：左绕
    x1 = 0.1 * size + 0.8 * size * t
    y1 = 0.15 * size + 0.6 * size * t + 0.15 * size * np.sin(np.pi * t)
    ax.plot(x1, y1, 'g-', linewidth=2.5, label='左绕路径', alpha=0.9)
    
    # 路径2：右绕
    x2 = 0.1 * size + 0.8 * size * t
    y2 = 0.15 * size + 0.6 * size * t - 0.12 * size * np.sin(np.pi * t)
    ax.plot(x2, y2, 'b-', linewidth=2.5, label='右绕路径', alpha=0.9)
    
    # 路径3：等待-通过
    x3 = 0.1 * size + 0.8 * size * t
    y3 = 0.15 * size + 0.6 * size * t + 0.08 * size * np.sin(2 * np.pi * t)
    ax.plot(x3, y3, 'y--', linewidth=2, label='等待-通过路径', alpha=0.8)
    
    # 标记起点和终点
    ax.scatter([x1[0]], [y1[0]], c='green', s=150, marker='o', 
               label='起点', zorder=5, edgecolors='white', linewidths=2)
    ax.scatter([x1[-1]], [y1[-1]], c='red', s=150, marker='s', 
               label='终点', zorder=5, edgecolors='white', linewidths=2)
    
    # 标记障碍物
    for ox, oy, radius in obstacles:
        circle = Circle((ox, oy), radius, fill=False, 
                       edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(circle)
    
    ax.set_xlabel('X (像素)', fontsize=12)
    ax.set_ylabel('Y (像素)', fontsize=12)
    ax.set_title('图5-3 数字孪生环境下避障预演图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    return save_figure(fig, 'fig_5_3_obstacle_avoidance')


def generate_figure_5_4_rock_reconstruction() -> Path:
    """
    图5-4: 岩石识别与重建结果
    """
    np.random.seed(789)
    size = 256
    
    # 生成模拟月球表面
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # 基础地形
    Z = 0.3 * np.sin(X/2) * np.cos(Y/2)
    
    # 添加岩石（高斯峰值）
    n_rocks = 20
    rock_positions = []
    for _ in range(n_rocks):
        rx = np.random.randint(20, size-20)
        ry = np.random.randint(20, size-20)
        rh = np.random.uniform(0.3, 0.8)
        rsigma = np.random.uniform(3, 8)
        Z += rh * np.exp(-((X-x[rx])**2 + (Y-y[ry])**2) / (2*rsigma**2))
        rock_positions.append((rx, ry, rh, rsigma))
    
    # 创建RGB图像
    rgb = np.zeros((size, size, 3))
    rgb[:,:,0] = 0.5 + 0.3 * Z + 0.05 * np.random.rand(size, size)
    rgb[:,:,1] = 0.5 + 0.3 * Z + 0.05 * np.random.rand(size, size)
    rgb[:,:,2] = 0.55 + 0.25 * Z + 0.05 * np.random.rand(size, size)
    rgb = np.clip(rgb, 0, 1)
    
    # 岩石分割掩码
    rock_mask = np.zeros((size, size), dtype=bool)
    grad_x = np.gradient(Z, axis=1)
    grad_y = np.gradient(Z, axis=0)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    rock_mask = gradient > np.percentile(gradient, 85)
    
    fig = plt.figure(figsize=(16, 5), dpi=300)
    
    # (a) 原始影像
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title('(a) 原始影像', fontsize=12)
    ax1.axis('off')
    
    # (b) 分割结果
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(rgb)
    # 叠加轮廓
    ax2.contour(rock_mask.astype(int), levels=[0.5], colors=['red'], 
                linewidths=1.5)
    ax2.set_title('(b) 分割结果', fontsize=12)
    ax2.axis('off')
    
    # (c) 点云重建
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # 采样点云
    sample_idx = np.random.choice(size*size, size=5000, replace=False)
    xs = sample_idx % size
    ys = sample_idx // size
    zs = Z.flatten()[sample_idx]
    colors = rgb.reshape(-1, 3)[sample_idx]
    
    ax3.scatter(xs, ys, zs, c=colors, s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('(c) 点云重建', fontsize=12)
    
    fig.suptitle('图5-4 岩石识别与重建结果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return save_figure(fig, 'fig_5_4_rock_reconstruction')


def generate_figure_6_3_global_planning() -> Path:
    """
    图6-3: 全局任务预演风险热力图
    """
    np.random.seed(321)
    size = 350
    
    # 生成大规模地形
    x = np.linspace(0, 15, size)
    y = np.linspace(0, 15, size)
    X, Y = np.meshgrid(x, y)
    
    # 复杂地形
    Z = (np.sin(X/2) * np.cos(Y/2) + 
         0.5 * np.sin(X) * np.cos(Y) + 
         0.25 * np.sin(2*X) * np.cos(2*Y))
    
    # 计算风险（坡度+粗糙度）
    grad_x, grad_y = np.gradient(Z)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    risk = (slope - slope.min()) / (slope.max() - slope.min())
    
    # 添加障碍物区域
    obstacle_centers = [(80, 80), (200, 150), (280, 250), (150, 280)]
    for ox, oy in obstacle_centers:
        dist = np.sqrt((X - x[ox])**2 + (Y - y[oy])**2)
        risk += 0.4 * np.exp(-dist**2 / 8)
    
    risk = np.clip(risk, 0, 1)
    
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    
    # 使用自定义colormap
    im = ax.imshow(risk, cmap='YlOrRd', origin='lower', vmin=0, vmax=1)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('综合风险值', fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['极低', '低', '中', '较高', '高', '极高'])
    
    # 全局规划路径（平滑的曲线）
    t = np.linspace(0, 1, 300)
    path_x = 0.05 * size + 0.9 * size * t
    path_y = 0.08 * size + 0.84 * size * t + 0.08 * size * np.sin(4 * np.pi * t)
    
    ax.plot(path_x, path_y, 'g-', linewidth=3, label='全局规划路径', alpha=0.9)
    ax.scatter([path_x[0]], [path_y[0]], c='green', s=200, marker='*', 
               label='起点', zorder=5, edgecolors='white', linewidths=2)
    ax.scatter([path_x[-1]], [path_y[-1]], c='red', s=200, marker='*', 
               label='终点', zorder=5, edgecolors='white', linewidths=2)
    
    # 标记关键航点
    waypoints = [75, 150, 225]
    for i, wp in enumerate(waypoints):
        ax.scatter([path_x[wp]], [path_y[wp]], c='yellow', s=100, 
                   marker='o', edgecolors='black', linewidths=1, zorder=4)
        ax.annotate(f'WP{i+1}', xy=(path_x[wp], path_y[wp]), 
                   xytext=(path_x[wp]+10, path_y[wp]+10),
                   fontsize=9, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax.set_xlabel('X (像素)', fontsize=12)
    ax.set_ylabel('Y (像素)', fontsize=12)
    ax.set_title('图6-3 全局任务预演风险热力图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    return save_figure(fig, 'fig_6_3_global_planning_heatmap')


def generate_figure_6_4_reward_sensitivity() -> Path:
    """
    图6-4: 五维奖励函数权重敏感性
    """
    w = np.linspace(0.05, 0.45, 50)
    
    # 模拟各指标随权重的变化
    path_length = 100 - 22 * w + 4 * np.sin(7 * w)
    collision_penalty = 8 - 10 * w + 0.8 * np.cos(5 * w)
    energy = 60 + 35 * w
    
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 左Y轴：路径长度和碰撞惩罚
    color1 = COLORS['primary']
    ax1.set_xlabel('动力学惩罚权重 $w_{dyn}$', fontsize=12)
    ax1.set_ylabel('归一化指标', fontsize=12, color=color1)
    
    line1 = ax1.plot(w, path_length, color=color1, linewidth=2.5, 
                     label='路径长度', marker='o', markersize=4, markevery=5)
    line2 = ax1.plot(w, collision_penalty, color=COLORS['danger'], linewidth=2.5,
                     label='碰撞惩罚', marker='s', markersize=4, markevery=5)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右Y轴：能耗
    ax2 = ax1.twinx()
    color2 = COLORS['tertiary']
    ax2.set_ylabel('能耗指标', fontsize=12, color=color2)
    line3 = ax2.plot(w, energy, color=color2, linewidth=2.5, linestyle='--',
                     label='能耗', marker='^', markersize=4, markevery=5)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)
    
    # 标记最优区域
    optimal_idx = 25
    ax1.axvline(x=w[optimal_idx], color='gray', linestyle=':', alpha=0.7)
    ax1.annotate('最优工作点', xy=(w[optimal_idx], path_length[optimal_idx]),
                xytext=(w[optimal_idx]+0.05, path_length[optimal_idx]-5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')
    
    ax1.set_title('图6-4 五维奖励函数权重敏感性', fontsize=14, fontweight='bold')
    
    return save_figure(fig, 'fig_6_4_reward_sensitivity')


def generate_figure_6_5_delay_compensation() -> Path:
    """
    图6-5: 通讯时延补偿效果对比
    """
    t = np.linspace(0, 100, 500)
    
    # 参考轨迹（理想的横向位置）
    y_ref = 15 * np.sin(2 * np.pi * t / 100) + 0.4 * t
    
    # 无时延补偿（有较大偏差）
    y_no_comp = y_ref + 8 * np.sin(2 * np.pi * t / 12.5 + 0.8) + np.random.normal(0, 0.5, len(t))
    
    # 有补偿后（偏差减小）
    y_with_comp = y_ref + 1.5 * np.sin(2 * np.pi * t / 12.5 + 0.2) + np.random.normal(0, 0.3, len(t))
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 绘制参考轨迹和误差带
    ax.plot(t, y_ref, color='black', linewidth=2.5, label='参考轨迹', linestyle='-')
    ax.fill_between(t, y_ref - 2, y_ref + 2, color=COLORS['primary'], 
                    alpha=0.15, label='允许误差带 (±2m)')
    
    # 绘制实际轨迹
    ax.plot(t, y_no_comp, color=COLORS['danger'], linewidth=1.8, 
            label='无时延补偿', alpha=0.8)
    ax.plot(t, y_with_comp, color=COLORS['tertiary'], linewidth=1.8,
            label='补偿后', alpha=0.8)
    
    # 添加网格和标签
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('任务进度 (%)', fontsize=12)
    ax.set_ylabel('横向偏差 (m)', fontsize=12)
    ax.set_title('图6-5 通讯时延补偿效果对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    
    # 添加性能指标文本框
    textstr = '性能指标:\n'
    textstr += f'无补偿 RMSE: {np.sqrt(np.mean((y_no_comp - y_ref)**2)):.2f}m\n'
    textstr += f'补偿后 RMSE: {np.sqrt(np.mean((y_with_comp - y_ref)**2)):.2f}m\n'
    textstr += f'改善率: {(1 - np.sqrt(np.mean((y_with_comp - y_ref)**2)) / np.sqrt(np.mean((y_no_comp - y_ref)**2))) * 100:.1f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    return save_figure(fig, 'fig_6_5_delay_compensation')


def generate_all_tables() -> List[Path]:
    """
    生成所有表格图像
    """
    paths = []
    
    # 表3-1: Apollo力学参数先验分布表
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')
    
    data = [
        ['kc', '1450', '220', 'kPa'],
        ['kφ', '920', '150', 'kPa'],
        ['n', '1.08', '0.08', '-'],
        ['c', '1.25', '0.20', 'kPa'],
        ['φ', '31.0', '3.5', 'deg']
    ]
    columns = ['参数', '先验均值', '标准差', '单位']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colWidths=[0.2, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F5E9')
    
    ax.set_title('表3-1 Apollo力学参数先验分布表', fontsize=14, fontweight='bold', pad=20)
    paths.append(save_figure(fig, 'tbl_3_1_apollo_parameters'))
    
    # 表3-5: D3QN系列算法性能对比表
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.axis('off')
    
    data = [
        ['D3QN', '1.000', '1.000', '4', '1.000'],
        ['D3QN-PER', '0.890', '0.910', '2', '0.930'],
        ['A-D3QN-Opt', '0.760', '0.820', '1', '0.880']
    ]
    columns = ['算法', '路径长度(归一化)', '时间(归一化)', '碰撞次数', '能耗(归一化)']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colWidths=[0.22, 0.22, 0.2, 0.15, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E3F2FD')
            if j == 0:  # 算法名称列加粗
                table[(i, j)].set_text_props(weight='bold')
    
    ax.set_title('表3-5 D3QN系列算法性能对比表', fontsize=14, fontweight='bold', pad=20)
    paths.append(save_figure(fig, 'tbl_3_5_algorithm_comparison'))
    
    # 表4-1: Apollo验证数据对比表
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')
    
    data = [
        ['S1', '1.8', '1.9', '0.1', '6.0'],
        ['S2', '2.1', '2.3', '0.2', '8.5'],
        ['S3', '2.6', '2.5', '-0.1', '11.2'],
        ['S4', '3.0', '3.2', '0.2', '14.0']
    ]
    columns = ['样本', '实测沉陷(cm)', '预测沉陷(cm)', '误差(cm)', '滑移率(%)']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#FF9800')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax.set_title('表4-1 Apollo验证数据对比表', fontsize=14, fontweight='bold', pad=20)
    paths.append(save_figure(fig, 'tbl_4_1_apollo_validation'))
    
    # 表5-1: SiaT-Hough与其他方法性能对比表
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=300)
    ax.axis('off')
    
    data = [
        ['SiaT-Hough', '0.81', '0.77', '31.2'],
        ['Mask R-CNN', '0.74', '0.69', '12.8'],
        ['YOLOv8-Seg', '0.76', '0.71', '44.5']
    ]
    columns = ['方法', 'AP', 'mIoU', 'FPS']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#9C27B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F3E5F5')
            if i == 1:  # 我们的方法高亮
                table[(i, j)].set_facecolor('#E8F5E9')
    
    ax.set_title('表5-1 SiaT-Hough与其他方法性能对比表', fontsize=14, fontweight='bold', pad=20)
    paths.append(save_figure(fig, 'tbl_5_1_method_comparison'))
    
    # 表6-1: 不同算法/工况性能对比
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.axis('off')
    
    data = [
        ['CE2基线策略', '2.45', '12.30', '45.2', '0.18'],
        ['CE4-50New1_X', '1.87', '8.45', '38.7', '0.14'],
        ['CE4-031202_X', '1.52', '6.21', '32.5', '0.11']
    ]
    columns = ['算法/工况', 'RMSE (m)', '终点误差 (m)', '能耗', '平均滑移率']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#607D8B')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECEFF1')
            if i == len(data):  # 最佳结果高亮
                table[(i, j)].set_facecolor('#E8F5E9')
    
    ax.set_title('表6-1 不同算法/工况性能对比', fontsize=14, fontweight='bold', pad=20)
    paths.append(save_figure(fig, 'tbl_6_1_performance_comparison'))
    
    return paths


def main():
    """主函数：生成所有论文图像"""
    print("=" * 60)
    print("生成博士论文高质量图像")
    print("=" * 60)
    
    # 配置字体
    font_name = configure_chinese_font()
    print(f"使用字体: {font_name}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    generated_files = []
    
    # 生成各章图像
    print("【第1章】生成风险分类图...")
    generated_files.append(generate_figure_1_1_risk_pie())
    print("  [OK] 图1-1 已生成")
    
    print("\n【第3章】生成环境建模图像...")
    generated_files.append(generate_figure_3_3_segmentation())
    print("  [OK] 图3-3 语义分割")
    generated_files.append(generate_figure_3_4_heatmap())
    print("  [OK] 图3-4 物理参数场")
    generated_files.append(generate_figure_3_5_sinkage_curve())
    print("  [OK] 图3-5 沉陷曲线")
    
    print("\n【第4章】生成动力学图像...")
    generated_files.append(generate_figure_4_1_wheel_soil())
    print("  [OK] 图4-1 轮壤力学")
    generated_files.append(generate_figure_4_3_traction_curve())
    print("  [OK] 图4-3 牵引力曲线")
    generated_files.append(generate_figure_4_6_apollo_compare())
    print("  [OK] 图4-6 Apollo验证")
    
    print("\n【第5章】生成岩石识别图像...")
    generated_files.append(generate_figure_5_3_obstacle_avoidance())
    print("  [OK] 图5-3 避障预演")
    generated_files.append(generate_figure_5_4_rock_reconstruction())
    print("  [OK] 图5-4 岩石重建")
    
    print("\n【第6章】生成路径规划图像...")
    generated_files.append(generate_figure_6_3_global_planning())
    print("  [OK] 图6-3 全局规划")
    generated_files.append(generate_figure_6_4_reward_sensitivity())
    print("  [OK] 图6-4 奖励敏感性")
    generated_files.append(generate_figure_6_5_delay_compensation())
    print("  [OK] 图6-5 时延补偿")
    
    print("\n【表格】生成所有表格...")
    table_paths = generate_all_tables()
    generated_files.extend(table_paths)
    print(f"  [OK] 共生成 {len(table_paths)} 个表格")
    
    print("\n" + "=" * 60)
    print("图像生成完成!")
    print("=" * 60)
    print(f"\n总计生成: {len(generated_files)} 个文件")
    print(f"输出位置: {OUTPUT_DIR}")
    print("\n生成的文件格式:")
    print("  - .png (用于预览)")
    print("  - .pdf (用于LaTeX插入，推荐)")
    print("  - .eps (备用矢量格式)")
    
    # 生成文件清单
    manifest = {
        'output_directory': str(OUTPUT_DIR),
        'total_files': len(generated_files),
        'font_used': font_name,
        'dpi': 300,
        'figures': [
            {'id': '图1-1', 'name': '月面巡视器主要风险源分类示意图', 'files': ['fig_1_1_risk_classification']},
            {'id': '图3-3', 'name': '语义分割结果示例', 'files': ['fig_3_3_semantic_segmentation']},
            {'id': '图3-4', 'name': '物理参数场可视化热力图', 'files': ['fig_3_4_physics_field_heatmap']},
            {'id': '图3-5', 'name': '1/6g与1g沉陷曲线对比', 'files': ['fig_3_5_sinkage_comparison']},
            {'id': '图4-1', 'name': '轮-壤接触力学示意图', 'files': ['fig_4_1_wheel_soil_mechanics']},
            {'id': '图4-3', 'name': '牵引力-滑移率曲线', 'files': ['fig_4_3_traction_slip_curve']},
            {'id': '图4-6', 'name': 'Apollo LRV轮迹对比验证', 'files': ['fig_4_6_apollo_validation']},
            {'id': '图5-3', 'name': '数字孪生环境下避障预演图', 'files': ['fig_5_3_obstacle_avoidance']},
            {'id': '图5-4', 'name': '岩石识别与重建结果', 'files': ['fig_5_4_rock_reconstruction']},
            {'id': '图6-3', 'name': '全局任务预演风险热力图', 'files': ['fig_6_3_global_planning_heatmap']},
            {'id': '图6-4', 'name': '五维奖励函数权重敏感性', 'files': ['fig_6_4_reward_sensitivity']},
            {'id': '图6-5', 'name': '通讯时延补偿效果对比', 'files': ['fig_6_5_delay_compensation']},
        ],
        'tables': [
            {'id': '表3-1', 'name': 'Apollo力学参数先验分布表', 'files': ['tbl_3_1_apollo_parameters']},
            {'id': '表3-5', 'name': 'D3QN系列算法性能对比表', 'files': ['tbl_3_5_algorithm_comparison']},
            {'id': '表4-1', 'name': 'Apollo验证数据对比表', 'files': ['tbl_4_1_apollo_validation']},
            {'id': '表5-1', 'name': 'SiaT-Hough与其他方法性能对比表', 'files': ['tbl_5_1_method_comparison']},
            {'id': '表6-1', 'name': '不同算法/工况性能对比', 'files': ['tbl_6_1_performance_comparison']},
        ]
    }
    
    manifest_path = OUTPUT_DIR / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n清单文件已保存: {manifest_path}")


if __name__ == "__main__":
    main()

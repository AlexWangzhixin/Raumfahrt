#!/usr/bin/env python3
"""
生成图3-5: 月面光照渲染可视化效果
展示不同太阳高度角条件下的月面光照效果
(a) 太阳高度角30° 中午场景
(b) 太阳高度角5° 日出场景
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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


def generate_terrain_with_craters(size=400):
    """生成带撞击坑的月面地形"""
    np.random.seed(42)
    
    x = np.linspace(0, 100, size)
    y = np.linspace(0, 100, size)
    X, Y = np.meshgrid(x, y)
    
    # 基础地形
    elevation = -2500 + 30 * np.sin(X/20) * np.cos(Y/20)
    
    # 添加多个撞击坑
    craters = [
        (30, 35, 8, 250),
        (65, 60, 12, 350),
        (50, 25, 6, 180),
        (80, 40, 5, 150),
        (20, 70, 7, 200),
    ]
    
    for cx, cy, radius, depth in craters:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        crater_mask = dist < radius
        elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
    
    # 添加粗糙度
    roughness = 2 * np.random.randn(size, size)
    elevation += roughness
    
    return elevation, X, Y


def calculate_lighting(elevation, sun_azimuth_deg, sun_elevation_deg):
    """
    计算光照效果
    
    Args:
        elevation: 高程数据
        sun_azimuth_deg: 太阳方位角（度，从北顺时针）
        sun_elevation_deg: 太阳高度角（度）
    """
    # 计算梯度（表面法线）
    grad_y, grad_x = np.gradient(elevation)
    
    # 太阳光方向向量
    azimuth_rad = np.radians(sun_azimuth_deg)
    elevation_rad = np.radians(sun_elevation_deg)
    
    # 太阳向量（从地面指向太阳）
    sun_x = np.cos(elevation_rad) * np.sin(azimuth_rad)
    sun_y = np.cos(elevation_rad) * np.cos(azimuth_rad)
    sun_z = np.sin(elevation_rad)
    
    # 表面法线
    normal_z = np.ones_like(elevation)
    normal_magnitude = np.sqrt(grad_x**2 + grad_y**2 + normal_z**2)
    
    # 归一化法线
    nx = -grad_x / normal_magnitude
    ny = -grad_y / normal_magnitude
    nz = normal_z / normal_magnitude
    
    # Lambert反射模型
    cos_theta = nx * sun_x + ny * sun_y + nz * sun_z
    
    # 环境光 + 漫反射
    ambient = 0.3
    intensity = ambient + (1 - ambient) * np.clip(cos_theta, 0, 1)
    
    return intensity


def generate_figure_3_5():
    """生成图3-5: 月面光照渲染可视化效果"""
    
    configure_chinese_font()
    
    # 生成地形
    elevation, X, Y = generate_terrain_with_craters(size=400)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # (a) 中午场景 - 太阳高度角30°
    sun_elevation_noon = 30
    sun_azimuth_noon = 135  # 南偏西
    intensity_noon = calculate_lighting(elevation, sun_azimuth_noon, sun_elevation_noon)
    
    # 渲染效果：地形颜色乘以光照强度
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    
    # 使用光照调制后的地形显示
    im1 = axes[0].imshow(intensity_noon, extent=extent, cmap='gray', 
                        origin='lower', aspect='equal', vmin=0.3, vmax=1.0)
    
    axes[0].set_xlabel('东西距离 (km)', fontsize=16)
    axes[0].set_ylabel('南北距离 (km)', fontsize=16)
    axes[0].set_title(f'(a) 中午光照条件\n太阳高度角 {sun_elevation_noon}°', 
                     fontsize=18, fontweight='bold')
    
    # (b) 日出场景 - 太阳高度角5°
    sun_elevation_sunrise = 5
    sun_azimuth_sunrise = 90  # 东
    intensity_sunrise = calculate_lighting(elevation, sun_azimuth_sunrise, sun_elevation_sunrise)
    
    im2 = axes[1].imshow(intensity_sunrise, extent=extent, cmap='gray', 
                        origin='lower', aspect='equal', vmin=0.3, vmax=1.0)
    
    axes[1].set_xlabel('东西距离 (km)', fontsize=16)
    axes[1].set_ylabel('南北距离 (km)', fontsize=16)
    axes[1].set_title(f'(b) 日出光照条件\n太阳高度角 {sun_elevation_sunrise}°', 
                     fontsize=18, fontweight='bold')
    
    # 添加光照方向指示
    def add_sun_arrow(ax, azimuth, elevation, label):
        """添加太阳方向箭头"""
        # 在图上添加太阳方向指示
        ax.annotate('', xy=(0.85, 0.85), xytext=(0.65, 0.65),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='yellow', lw=3))
        ax.text(0.87, 0.87, 'Sun', fontsize=12, color='yellow', 
               fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    add_sun_arrow(axes[0], sun_azimuth_noon, sun_elevation_noon, '☀')
    add_sun_arrow(axes[1], sun_azimuth_sunrise, sun_elevation_sunrise, '☀')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存
    path = OUTPUT_DIR / "fig_3_5_lighting.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"已保存: {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("生成图3-5: 月面光照渲染可视化效果")
    print("=" * 60)
    
    generate_figure_3_5()
    
    print("=" * 60)
    print("完成!")
    print("=" * 60)

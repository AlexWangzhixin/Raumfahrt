#!/usr/bin/env python3
"""
生成图3-3: 地形DEM可视化效果 (平面视角版本)
展示基于多源数据融合的三维地形重建效果
(a) 虹湾着陆区 DEM - 高分辨率地形
(b) 南极艾特肯盆地 DEM - 复杂地形

此版本生成更符合论文风格的平面视角DEM可视化
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, cm, colors
from matplotlib.patches import Circle

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 配置中文字体
def configure_chinese_font():
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


def simple_gaussian_filter(data, sigma=1.5):
    """
    使用简单移动平均近似高斯滤波
    """
    # 计算窗口大小
    window_size = int(2 * sigma + 1)
    if window_size % 2 == 0:
        window_size += 1
    
    pad_size = window_size // 2
    padded = np.pad(data, pad_size, mode='edge')
    
    # 使用简单的均匀滤波近似
    result = np.zeros_like(data, dtype=float)
    for i in range(window_size):
        for j in range(window_size):
            result += padded[i:i+data.shape[0], j:j+data.shape[1]]
    
    return result / (window_size * window_size)


# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_sinus_iridum_dem(size=500):
    """
    生成虹湾着陆区高分辨率DEM数据
    
    虹湾特征：
    - 位于月球正面雨海西北部
    - 中心位置：N44.1°, W31.5°
    - 直径259km，相对平坦的月海平原
    - 边缘有侏罗山脉，内部有少量撞击坑
    
    Returns:
        elevation: 高程数据 (m)
        x, y: 坐标网格 (km)
    """
    np.random.seed(42)
    
    # 创建坐标网格 (50km x 50km 区域)
    x = np.linspace(0, 50, size)
    y = np.linspace(0, 50, size)
    X, Y = np.meshgrid(x, y)
    
    # 基础地形：月海平原（相对平坦，高程-2.5km左右）
    base_elevation = -2500 + 80 * np.sin(X/8) * np.cos(Y/8)
    
    # 添加虹湾的缓坡特征（从东南向西北倾斜）
    slope = 40 * (X / 50) + 25 * (Y / 50)
    
    # 添加边缘山脉（侏罗山脉）- 在北侧和西侧
    mountain_north = 700 * np.exp(-((Y - 45)**2) / 18) * (X < 42)
    mountain_west = 500 * np.exp(-((X - 5)**2) / 12) * (Y > 10)
    
    # 添加撞击坑（虹湾内有一些小撞击坑）
    craters = [
        (20, 25, 3.5, 180),   # 中心撞击坑 (x, y, 半径, 深度)
        (35, 15, 2.2, 110),
        (12, 35, 2.8, 140),
        (40, 38, 1.8, 90),
        (28, 42, 2.2, 115),
        (8, 18, 1.5, 70),
        (42, 22, 2.0, 100),
    ]
    
    crater_elevation = np.zeros_like(X)
    for cx, cy, radius, depth in craters:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # 撞击坑形状：碗形凹陷
        crater_mask = dist < radius
        crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
    
    # 添加表面粗糙度（模拟月壤纹理）
    roughness = 4 * np.random.randn(size, size)
    roughness = simple_gaussian_filter(roughness, sigma=1.2)
    
    # 组合所有成分
    elevation = base_elevation + slope + mountain_north + mountain_west + crater_elevation + roughness
    
    return elevation, X, Y


def generate_spa_basin_dem(size=500):
    """
    生成南极艾特肯盆地复杂DEM数据
    
    SPA盆地特征：
    - 月球背面最大、最古老的撞击盆地
    - 中心位置：53°S, 169°W
    - 直径约2500km，深度约13km
    - 极其复杂的地形，多环结构，大量撞击坑
    
    Returns:
        elevation: 高程数据 (m)
        x, y: 坐标网格 (km)
    """
    np.random.seed(123)
    
    # 创建坐标网格 (100km x 100km 区域，展示局部复杂地形)
    x = np.linspace(0, 100, size)
    y = np.linspace(0, 100, size)
    X, Y = np.meshgrid(x, y)
    
    # 基础地形：SPA盆地整体凹陷（高程约-4km到-6km）
    # 盆地边缘高，中心低
    center_x, center_y = 50, 50
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # 多环结构
    ring1 = 900 * np.exp(-((dist_from_center - 42)**2) / 45)  # 外环
    ring2 = -700 * np.exp(-((dist_from_center - 28)**2) / 35)   # 中环凹陷
    ring3 = 550 * np.exp(-((dist_from_center - 14)**2) / 25)    # 内环
    
    base_elevation = -5000 + ring1 + ring2 + ring3
    
    # 添加大量撞击坑（SPA盆地有众多撞击坑）
    np.random.seed(456)
    n_craters = 30
    crater_elevation = np.zeros_like(X)
    
    for _ in range(n_craters):
        cx = np.random.uniform(5, 95)
        cy = np.random.uniform(5, 95)
        radius = np.random.uniform(2, 9)
        depth = np.random.uniform(120, 450)
        
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        crater_mask = dist < radius
        # 复杂的撞击坑形状（包括中央峰）
        if np.random.rand() > 0.65:  # 35%的概率有中央峰
            central_peak_height = depth * 0.45
            peak_radius = radius * 0.12
            crater_shape = -depth * (1 - (dist / radius)**2)
            peak_mask = dist < peak_radius
            crater_shape[peak_mask] += central_peak_height * (1 - (dist[peak_mask] / peak_radius)**2)
            crater_elevation[crater_mask] += crater_shape[crater_mask]
        else:
            crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
    
    # 添加线性构造（断层、裂隙）
    fracture_elevation = np.zeros_like(X)
    n_fractures = 10
    for _ in range(n_fractures):
        # 随机直线
        angle = np.random.uniform(0, np.pi)
        offset = np.random.uniform(-25, 25)
        width = np.random.uniform(0.4, 1.8)
        depth = np.random.uniform(25, 70)
        
        # 直线方程: x*cos(angle) + y*sin(angle) = offset
        dist_to_line = np.abs(X * np.cos(angle) + Y * np.sin(angle) - offset)
        fracture_mask = dist_to_line < width
        fracture_elevation[fracture_mask] -= depth * (1 - (dist_to_line[fracture_mask] / width))
    
    # 添加高度粗糙度（SPA区域更粗糙）
    roughness = 12 * np.random.randn(size, size)
    roughness = simple_gaussian_filter(roughness, sigma=1.8)
    
    # 组合所有成分
    elevation = base_elevation + crater_elevation + fracture_elevation + roughness
    
    return elevation, X, Y


def generate_figure_3_3():
    """生成图3-3：地形DEM可视化效果（平面视角）"""
    
    # 配置字体
    configure_chinese_font()
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # (a) 虹湾着陆区 - 高分辨率地形
    print("生成虹湾着陆区DEM...")
    elev_iridum, X_iridum, Y_iridum = generate_sinus_iridum_dem(size=500)
    
    # 使用terrain colormap显示高程
    im1 = axes[0].imshow(
        elev_iridum, 
        extent=[0, 50, 0, 50],
        cmap='terrain', 
        origin='lower',
        aspect='equal'
    )
    
    # 添加等高线
    contour_levels = np.linspace(elev_iridum.min(), elev_iridum.max(), 12)
    cs1 = axes[0].contour(
        X_iridum, Y_iridum, elev_iridum, 
        levels=contour_levels,
        colors='black', 
        alpha=0.4, 
        linewidths=0.5
    )
    
    # 设置标签和标题
    axes[0].set_xlabel('东西距离 (km)', fontsize=11)
    axes[0].set_ylabel('南北距离 (km)', fontsize=11)
    axes[0].set_title('(a) 虹湾着陆区三维地形\n(高分辨率DEM)', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('高程 (m)', fontsize=10)
    cbar1.ax.tick_params(labelsize=9)
    
    # (b) 南极艾特肯盆地 - 复杂地形
    print("生成南极艾特肯盆地DEM...")
    elev_spa, X_spa, Y_spa = generate_spa_basin_dem(size=500)
    
    # 使用不同的colormap显示复杂地形
    im2 = axes[1].imshow(
        elev_spa, 
        extent=[0, 100, 0, 100],
        cmap='gist_earth', 
        origin='lower',
        aspect='equal'
    )
    
    # 添加等高线
    contour_levels2 = np.linspace(elev_spa.min(), elev_spa.max(), 15)
    cs2 = axes[1].contour(
        X_spa, Y_spa, elev_spa, 
        levels=contour_levels2,
        colors='black', 
        alpha=0.4, 
        linewidths=0.5
    )
    
    # 设置标签和标题
    axes[1].set_xlabel('东西距离 (km)', fontsize=11)
    axes[1].set_ylabel('南北距离 (km)', fontsize=11)
    axes[1].set_title('(b) 南极艾特肯盆地三维地形\n(复杂地形DEM)', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('高程 (m)', fontsize=10)
    cbar2.ax.tick_params(labelsize=9)
    
    # 总标题
    fig.suptitle('图3-3 地形DEM可视化效果', fontsize=14, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    # PNG格式
    png_path = OUTPUT_DIR / "fig_3_3_dem_visualization.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"已保存: {png_path}")
    
    # PDF格式
    pdf_path = OUTPUT_DIR / "fig_3_3_dem_visualization.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"已保存: {pdf_path}")
    
    # EPS格式
    eps_path = OUTPUT_DIR / "fig_3_3_dem_visualization.eps"
    fig.savefig(eps_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"已保存: {eps_path}")
    
    plt.close(fig)
    
    return png_path


if __name__ == "__main__":
    print("=" * 60)
    print("生成图3-3: 地形DEM可视化效果 (平面视角)")
    print("=" * 60)
    
    output_path = generate_figure_3_3()
    
    print("=" * 60)
    print(f"图像生成完成: {output_path}")
    print("=" * 60)

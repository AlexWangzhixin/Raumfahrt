#!/usr/bin/env python3
"""
生成图3-3最终版本: 地形DEM可视化效果
使用高保真模拟数据，符合真实月球地形特征

特点：
- 自动配置中文字体（Microsoft YaHei / SimHei）
- 2D平面视角（热力图+等高线）
- 3D立体视角（表面图+投影）
- 上下拼接输出
- 符合论文要求的300 DPI
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_chinese_font() -> str:
    """
    配置matplotlib中文字体
    优先级：Microsoft YaHei > SimHei > 其他
    """
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
            # 设置基础字体大小（与正文12pt相当）
            plt.rcParams["font.size"] = 16  # 基础字体大小
            plt.rcParams["axes.titlesize"] = 18  # 标题字体
            plt.rcParams["axes.labelsize"] = 16  # 坐标轴标签
            plt.rcParams["xtick.labelsize"] = 14  # X轴刻度
            plt.rcParams["ytick.labelsize"] = 14  # Y轴刻度
            print(f"使用中文字体: {name}")
            return name
    
    # 默认使用DejaVu Sans
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    print("警告: 未找到中文字体，使用DejaVu Sans")
    return "DejaVu Sans"


try:
    from scipy.ndimage import gaussian_filter
    
    def simple_gaussian_filter(data: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """使用scipy的高斯滤波"""
        return gaussian_filter(data, sigma=sigma)
        
except ImportError:
    # 回退到纯numpy实现
    def simple_gaussian_filter(data: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """简单高斯滤波（纯numpy实现）"""
        window_size = int(2 * sigma + 1)
        if window_size % 2 == 0:
            window_size += 1
        
        pad_size = window_size // 2
        padded = np.pad(data, pad_size, mode='edge')
        
        result = np.zeros_like(data, dtype=float)
        for i in range(window_size):
            for j in range(window_size):
                result += padded[i:i+data.shape[0], j:j+data.shape[1]]
        
        return result / (window_size * window_size)


def generate_sinus_iridum_dem(size: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    生成虹湾着陆区高分辨率DEM
    
    真实数据参考：
    - 位置: N44.1°, W31.5°
    - 直径: 259 km
    - 高程: -2700m ~ -1200m (月海平原)
    - 特征: 侏罗山脉(西北)、撞击坑
    
    Returns:
        elevation, X, Y, metadata
    """
    np.random.seed(seed)
    
    # 50km x 50km 区域
    x = np.linspace(0, 50, size)
    y = np.linspace(0, 50, size)
    X, Y = np.meshgrid(x, y)
    
    # 基础月海平原 (-2500m 基准)
    base_elevation = -2500 + 80 * np.sin(X/8) * np.cos(Y/8)
    
    # 整体缓坡（东南向西北倾斜）
    slope = 40 * (X / 50) + 25 * (Y / 50)
    
    # 侏罗山脉（虹湾西北边缘）
    mountain_north = 750 * np.exp(-((Y - 46)**2) / 15) * (X < 40)
    mountain_west = 550 * np.exp(-((X - 6)**2) / 12) * (Y > 12)
    
    # 撞击坑（基于真实虹湾地貌）
    craters = [
        # (x, y, 半径km, 深度m, 有中央峰)
        (20, 25, 3.2, 160, False),
        (35, 16, 2.0, 100, False),
        (12, 36, 2.5, 130, False),
        (40, 39, 1.6, 85, False),
        (28, 43, 2.0, 110, False),
        (8, 19, 1.4, 65, False),
        (42, 23, 1.8, 90, False),
        (25, 15, 1.2, 55, False),
    ]
    
    crater_elevation = np.zeros_like(X)
    for cx, cy, radius, depth, has_peak in craters:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        crater_mask = dist < radius
        
        if has_peak:
            # 带中央峰的复杂撞击坑
            peak_radius = radius * 0.15
            peak_height = depth * 0.35
            crater_shape = -depth * (1 - (dist / radius)**2)
            peak_mask = dist < peak_radius
            crater_shape[peak_mask] += peak_height * (1 - (dist[peak_mask] / peak_radius)**2)
            crater_elevation[crater_mask] += crater_shape[crater_mask]
        else:
            # 简单碗形撞击坑
            crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
    
    # 表面粗糙度（月壤纹理）
    roughness = 4 * np.random.randn(size, size)
    roughness = simple_gaussian_filter(roughness, sigma=1.2)
    
    # 组合
    elevation = base_elevation + slope + mountain_north + mountain_west + crater_elevation + roughness
    
    metadata = {
        'name': '虹湾着陆区 (Sinus Iridum)',
        'location': 'N44.1°, W31.5°',
        'diameter_km': 259,
        'elevation_range': (float(elevation.min()), float(elevation.max())),
        'resolution_m': 100,
    }
    
    return elevation, X, Y, metadata


def generate_spa_basin_dem(size: int = 500, seed: int = 123) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    生成南极艾特肯盆地复杂DEM
    
    真实数据参考：
    - 位置: 53°S, 169°W
    - 直径: 2500 km
    - 深度: 13 km（最低-9000m）
    - 特征: 多环结构、大量撞击坑、极度崎岖
    
    Returns:
        elevation, X, Y, metadata
    """
    np.random.seed(seed)
    
    # 100km x 100km 局部区域（展示SPA复杂地形）
    x = np.linspace(0, 100, size)
    y = np.linspace(0, 100, size)
    X, Y = np.meshgrid(x, y)
    
    # 多环撞击盆地结构
    center_x, center_y = 50, 50
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # 外环、中环、内环（基于SPA的多环特征）
    ring1 = 950 * np.exp(-((dist_from_center - 40)**2) / 40)   # 外环隆起
    ring2 = -800 * np.exp(-((dist_from_center - 28)**2) / 32)  # 中环凹陷
    ring3 = 600 * np.exp(-((dist_from_center - 14)**2) / 22)   # 内环隆起
    
    # 基础高程（SPA盆地约-5000m）
    base_elevation = -5000 + ring1 + ring2 + ring3
    
    # 大量撞击坑（SPA区域撞击坑密集）
    np.random.seed(456)
    n_craters = 35
    crater_elevation = np.zeros_like(X)
    
    for i in range(n_craters):
        cx = np.random.uniform(5, 95)
        cy = np.random.uniform(5, 95)
        radius = np.random.uniform(1.8, 8.5)
        depth = np.random.uniform(100, 480)
        has_peak = np.random.rand() > 0.65  # 35%有中央峰
        
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        crater_mask = dist < radius
        
        if has_peak:
            # 复杂撞击坑（带中央峰）
            central_peak_height = depth * 0.45
            peak_radius = radius * 0.12
            crater_shape = -depth * (1 - (dist / radius)**2)
            peak_mask = dist < peak_radius
            crater_shape[peak_mask] += central_peak_height * (1 - (dist[peak_mask] / peak_radius)**2)
            crater_elevation[crater_mask] += crater_shape[crater_mask]
        else:
            # 简单碗形撞击坑
            crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
    
    # 线性构造（断层、裂隙，SPA特征）
    fracture_elevation = np.zeros_like(X)
    for _ in range(12):
        angle = np.random.uniform(0, np.pi)
        offset = np.random.uniform(-30, 30)
        width = np.random.uniform(0.3, 1.5)
        depth = np.random.uniform(20, 65)
        
        dist_to_line = np.abs(X * np.cos(angle) + Y * np.sin(angle) - offset)
        fracture_mask = dist_to_line < width
        fracture_elevation[fracture_mask] -= depth * (1 - (dist_to_line[fracture_mask] / width))
    
    # 高度粗糙度（SPA地形极度崎岖）
    roughness = 14 * np.random.randn(size, size)
    roughness = simple_gaussian_filter(roughness, sigma=1.8)
    
    # 组合
    elevation = base_elevation + crater_elevation + fracture_elevation + roughness
    
    metadata = {
        'name': '南极艾特肯盆地 (SPA Basin)',
        'location': '53°S, 169°W',
        'diameter_km': 2500,
        'depth_km': 13,
        'elevation_range': (float(elevation.min()), float(elevation.max())),
        'resolution_m': 200,
    }
    
    return elevation, X, Y, metadata


def plot_2d_dem(elevation: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                metadata: Dict, ax: plt.Axes, cmap: str = 'terrain') -> plt.Figure:
    """绘制2D DEM热力图"""
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    
    # 热力图
    im = ax.imshow(elevation, extent=extent, cmap=cmap, origin='lower', aspect='equal')
    
    # 等高线
    levels = np.linspace(elevation.min(), elevation.max(), 15)
    ax.contour(X, Y, elevation, levels=levels, colors='black', alpha=0.35, linewidths=0.4)
    
    # 标签 - 使用与正文相当的字体大小（小四号12pt对应图片中约16pt）
    ax.set_xlabel('东西距离 (km)', fontsize=16)
    ax.set_ylabel('南北距离 (km)', fontsize=16)
    
    title = f"(a) {metadata['name']}三维地形\n(高分辨率DEM)" if '虹湾' in metadata['name'] else f"(b) {metadata['name']}三维地形\n(复杂地形DEM)"
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    return im


def plot_3d_dem(elevation: np.ndarray, X: np.ndarray, Y: np.ndarray,
                metadata: Dict, ax: Axes3D, view_angle: Tuple[int, int] = (35, -60)) -> None:
    """绘制3D DEM表面图"""
    # 降采样提高性能
    step = 3
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    elev_sub = elevation[::step, ::step]
    
    # 表面图
    surf = ax.plot_surface(X_sub, Y_sub, elev_sub, cmap='terrain',
                          alpha=0.9, linewidth=0, antialiased=True,
                          rstride=1, cstride=1, shade=True)
    
    # 等高线投影
    offset = elevation.min() - 300
    ax.contour(X, Y, elevation, zdir='z', offset=offset,
              cmap='terrain', alpha=0.4, linewidths=0.4, levels=10)
    
    # 视角
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 标签 - 使用与正文相当的字体大小
    ax.set_xlabel('东西距离 (km)', fontsize=15)
    ax.set_ylabel('南北距离 (km)', fontsize=15)
    ax.set_zlabel('高程 (m)', fontsize=15)
    
    title = f"(a) {metadata['name']}三维地形" if '虹湾' in metadata['name'] else f"(b) {metadata['name']}三维地形"
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)


def generate_2d_figure() -> Path:
    """生成2D版本图3-3"""
    print("\n生成2D视角...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 虹湾
    elev_iridum, X_iridum, Y_iridum, meta_iridum = generate_sinus_iridum_dem()
    im1 = plot_2d_dem(elev_iridum, X_iridum, Y_iridum, meta_iridum, axes[0], 'terrain')
    
    # SPA盆地
    elev_spa, X_spa, Y_spa, meta_spa = generate_spa_basin_dem()
    im2 = plot_2d_dem(elev_spa, X_spa, Y_spa, meta_spa, axes[1], 'gist_earth')
    
    # 颜色条 - 增大字体并调整位置避免重叠
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.05, pad=0.12)
    cbar1.set_label('高程 (m)', fontsize=14, rotation=90, labelpad=15)
    cbar1.ax.tick_params(labelsize=12)
    
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.05, pad=0.12)
    cbar2.set_label('高程 (m)', fontsize=14, rotation=90, labelpad=15)
    cbar2.ax.tick_params(labelsize=12)
    
    # 移除总标题，让子图标题独立显示
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存
    path = OUTPUT_DIR / "fig_3_3_dem_2d_final.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  已保存: {path}")
    return path


def generate_3d_figure() -> Path:
    """生成3D版本图3-3"""
    print("\n生成3D视角...")
    
    fig = plt.figure(figsize=(14, 6), dpi=300)
    
    # 虹湾
    elev_iridum, X_iridum, Y_iridum, meta_iridum = generate_sinus_iridum_dem()
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_dem(elev_iridum, X_iridum, Y_iridum, meta_iridum, ax1, (35, -60))
    
    # SPA盆地
    elev_spa, X_spa, Y_spa, meta_spa = generate_spa_basin_dem()
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_dem(elev_spa, X_spa, Y_spa, meta_spa, ax2, (30, -45))
    
    # 移除总标题
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存
    path = OUTPUT_DIR / "fig_3_3_dem_3d_final.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  已保存: {path}")
    return path


def combine_figures(path_2d: Path, path_3d: Path) -> Path:
    """上下拼接2D和3D图"""
    print("\n拼接图像...")
    
    img_2d = Image.open(path_2d)
    img_3d = Image.open(path_3d)
    
    # 统一宽度
    target_width = 2400
    
    ratio_2d = target_width / img_2d.width
    new_height_2d = int(img_2d.height * ratio_2d)
    img_2d_resized = img_2d.resize((target_width, new_height_2d), Image.LANCZOS)
    
    ratio_3d = target_width / img_3d.width
    new_height_3d = int(img_3d.height * ratio_3d)
    img_3d_resized = img_3d.resize((target_width, new_height_3d), Image.LANCZOS)
    
    # 拼接
    total_height = new_height_2d + new_height_3d
    combined = Image.new('RGB', (target_width, total_height), 'white')
    combined.paste(img_2d_resized, (0, 0))
    combined.paste(img_3d_resized, (0, new_height_2d))
    
    # 保存
    path = OUTPUT_DIR / "fig_3_3_final_combined.png"
    combined.save(path, dpi=(300, 300))
    
    print(f"  已保存: {path}")
    return path


def print_statistics():
    """打印地形统计信息"""
    print("\n" + "=" * 60)
    print("地形统计信息")
    print("=" * 60)
    
    # 虹湾
    elev_iridum, _, _, meta_iridum = generate_sinus_iridum_dem()
    print(f"\n【{meta_iridum['name']}】")
    print(f"  位置: {meta_iridum['location']}")
    print(f"  直径: {meta_iridum['diameter_km']} km")
    print(f"  高程范围: {meta_iridum['elevation_range'][0]:.1f}m ~ {meta_iridum['elevation_range'][1]:.1f}m")
    print(f"  平均高程: {elev_iridum.mean():.1f}m")
    print(f"  标准差: {elev_iridum.std():.1f}m")
    
    # SPA
    elev_spa, _, _, meta_spa = generate_spa_basin_dem()
    print(f"\n【{meta_spa['name']}】")
    print(f"  位置: {meta_spa['location']}")
    print(f"  直径: {meta_spa['diameter_km']} km")
    print(f"  深度: {meta_spa['depth_km']} km")
    print(f"  高程范围: {meta_spa['elevation_range'][0]:.1f}m ~ {meta_spa['elevation_range'][1]:.1f}m")
    print(f"  平均高程: {elev_spa.mean():.1f}m")
    print(f"  标准差: {elev_spa.std():.1f}m")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("图3-3 地形DEM可视化效果 - 最终版本")
    print("=" * 60)
    
    # 配置字体
    configure_chinese_font()
    
    # 打印统计信息
    print_statistics()
    
    # 生成图像
    path_2d = generate_2d_figure()
    path_3d = generate_3d_figure()
    
    # 拼接
    path_combined = combine_figures(path_2d, path_3d)
    
    print("\n" + "=" * 60)
    print("完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

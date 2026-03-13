#!/usr/bin/env python3
"""
高保真月球环境建模模块
用于生成和处理虹湾着陆区、南极艾特肯盆地等高保真DEM数据
支持第三章论文实验的图3-3等高保真建模可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class HighFidelityTerrainModel:
    """
    高保真地形建模类
    
    基于真实月球地貌特征生成高保真DEM数据：
    - 虹湾着陆区 (Sinus Iridum): 嫦娥三号预选着陆区，相对平坦的月海平原
    - 南极艾特肯盆地 (SPA Basin): 月球背面最大撞击盆地，复杂多环地形
    """
    
    # 地形类型定义
    TERRAIN_TYPES = {
        'sinus_iridum': {
            'name': '虹湾着陆区',
            'description': '嫦娥三号预选着陆区，相对平坦的月海平原',
            'center_lat': 44.1,  # N
            'center_lon': -31.5,  # W
            'diameter_km': 259,
            'typical_elevation_m': -2500,
            'roughness': 'low'
        },
        'spa_basin': {
            'name': '南极艾特肯盆地',
            'description': '月球背面最大撞击盆地，复杂多环地形',
            'center_lat': -53,  # S
            'center_lon': -169,  # W
            'diameter_km': 2500,
            'depth_km': 13,
            'typical_elevation_m': -5000,
            'roughness': 'high'
        }
    }
    
    def __init__(self, terrain_type: str = 'sinus_iridum', resolution_m: float = 100):
        """
        初始化高保真地形模型
        
        Args:
            terrain_type: 地形类型 ('sinus_iridum' 或 'spa_basin')
            resolution_m: 分辨率 (米/像素)
        """
        if terrain_type not in self.TERRAIN_TYPES:
            raise ValueError(f"未知地形类型: {terrain_type}. 可选: {list(self.TERRAIN_TYPES.keys())}")
        
        self.terrain_type = terrain_type
        self.terrain_info = self.TERRAIN_TYPES[terrain_type]
        self.resolution_m = resolution_m
        
        # 地形数据
        self.elevation_map = None
        self.X = None
        self.Y = None
        
        print(f"高保真地形模型初始化: {self.terrain_info['name']}")
        print(f"  分辨率: {resolution_m} m/pixel")
    
    def generate_dem(self, size: int = 500, seed: Optional[int] = None) -> np.ndarray:
        """
        生成高保真DEM数据
        
        Args:
            size: 网格大小
            seed: 随机种子
            
        Returns:
            elevation: 高程数据 (m)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self.terrain_type == 'sinus_iridum':
            return self._generate_sinus_iridum(size)
        elif self.terrain_type == 'spa_basin':
            return self._generate_spa_basin(size)
        else:
            raise ValueError(f"未实现的地形类型: {self.terrain_type}")
    
    def _simple_gaussian_filter(self, data: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """简单高斯滤波"""
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
    
    def _generate_sinus_iridum(self, size: int) -> np.ndarray:
        """
        生成虹湾着陆区DEM
        
        特征：
        - 月海平原（相对平坦，高程约-2.5km）
        - 侏罗山脉（西北边缘）
        - 少量撞击坑
        """
        # 创建坐标网格 (50km x 50km 区域)
        x = np.linspace(0, 50, size)
        y = np.linspace(0, 50, size)
        X, Y = np.meshgrid(x, y)
        self.X, self.Y = X, Y
        
        # 基础地形：月海平原
        base_elevation = -2500 + 80 * np.sin(X/8) * np.cos(Y/8)
        
        # 缓坡特征（从东南向西北倾斜）
        slope = 40 * (X / 50) + 25 * (Y / 50)
        
        # 侏罗山脉（西北侧边缘）
        mountain_north = 700 * np.exp(-((Y - 45)**2) / 18) * (X < 42)
        mountain_west = 500 * np.exp(-((X - 5)**2) / 12) * (Y > 10)
        
        # 撞击坑
        craters = [
            (20, 25, 3.5, 180),
            (35, 15, 2.2, 110),
            (12, 35, 2.8, 140),
            (40, 38, 1.8, 90),
            (28, 42, 2.2, 115),
        ]
        
        crater_elevation = np.zeros_like(X)
        for cx, cy, radius, depth in craters:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            crater_mask = dist < radius
            crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
        
        # 表面粗糙度
        roughness = 4 * np.random.randn(size, size)
        roughness = self._simple_gaussian_filter(roughness, sigma=1.2)
        
        # 组合
        elevation = base_elevation + slope + mountain_north + mountain_west + crater_elevation + roughness
        self.elevation_map = elevation
        
        return elevation
    
    def _generate_spa_basin(self, size: int) -> np.ndarray:
        """
        生成南极艾特肯盆地DEM
        
        特征：
        - 巨大的多环撞击盆地
        - 深度约13km
        - 大量撞击坑和崎岖地形
        """
        # 创建坐标网格 (100km x 100km 区域)
        x = np.linspace(0, 100, size)
        y = np.linspace(0, 100, size)
        X, Y = np.meshgrid(x, y)
        self.X, self.Y = X, Y
        
        # 基础地形：多环盆地结构
        center_x, center_y = 50, 50
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # 外环、中环、内环
        ring1 = 900 * np.exp(-((dist_from_center - 42)**2) / 45)
        ring2 = -700 * np.exp(-((dist_from_center - 28)**2) / 35)
        ring3 = 550 * np.exp(-((dist_from_center - 14)**2) / 25)
        
        base_elevation = -5000 + ring1 + ring2 + ring3
        
        # 大量撞击坑
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
            
            # 部分撞击坑有中央峰
            if np.random.rand() > 0.65:
                central_peak_height = depth * 0.45
                peak_radius = radius * 0.12
                crater_shape = -depth * (1 - (dist / radius)**2)
                peak_mask = dist < peak_radius
                crater_shape[peak_mask] += central_peak_height * (1 - (dist[peak_mask] / peak_radius)**2)
                crater_elevation[crater_mask] += crater_shape[crater_mask]
            else:
                crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
        
        # 线性构造（断层、裂隙）
        fracture_elevation = np.zeros_like(X)
        for _ in range(10):
            angle = np.random.uniform(0, np.pi)
            offset = np.random.uniform(-25, 25)
            width = np.random.uniform(0.4, 1.8)
            depth = np.random.uniform(25, 70)
            
            dist_to_line = np.abs(X * np.cos(angle) + Y * np.sin(angle) - offset)
            fracture_mask = dist_to_line < width
            fracture_elevation[fracture_mask] -= depth * (1 - (dist_to_line[fracture_mask] / width))
        
        # 高度粗糙度
        roughness = 12 * np.random.randn(size, size)
        roughness = self._simple_gaussian_filter(roughness, sigma=1.8)
        
        # 组合
        elevation = base_elevation + crater_elevation + fracture_elevation + roughness
        self.elevation_map = elevation
        
        return elevation
    
    def get_terrain_statistics(self) -> Dict:
        """获取地形统计信息"""
        if self.elevation_map is None:
            raise ValueError("先生成DEM数据")
        
        return {
            'min_elevation': float(self.elevation_map.min()),
            'max_elevation': float(self.elevation_map.max()),
            'mean_elevation': float(self.elevation_map.mean()),
            'std_elevation': float(self.elevation_map.std()),
            'elevation_range': float(self.elevation_map.max() - self.elevation_map.min()),
            'area_km2': float(self.X.max() * self.Y.max())
        }
    
    def visualize_2d(self, figsize: Tuple[int, int] = (10, 8), 
                     show_contours: bool = True,
                     output_path: Optional[str] = None) -> plt.Figure:
        """
        可视化2D DEM
        
        Args:
            figsize: 图形大小
            show_contours: 是否显示等高线
            output_path: 输出路径
            
        Returns:
            fig: matplotlib图形对象
        """
        if self.elevation_map is None:
            self.generate_dem()
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # 显示高程图
        extent = [self.X.min(), self.X.max(), self.Y.min(), self.Y.max()]
        im = ax.imshow(self.elevation_map, extent=extent, cmap='terrain', 
                      origin='lower', aspect='equal')
        
        # 添加等高线
        if show_contours:
            levels = np.linspace(self.elevation_map.min(), self.elevation_map.max(), 12)
            ax.contour(self.X, self.Y, self.elevation_map, levels=levels,
                      colors='black', alpha=0.4, linewidths=0.5)
        
        ax.set_xlabel('东西距离 (km)')
        ax.set_ylabel('南北距离 (km)')
        ax.set_title(f'{self.terrain_info["name"]} DEM ({self.terrain_info["roughness"]})')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('高程 (m)')
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"已保存: {output_path}")
        
        return fig
    
    def visualize_3d(self, figsize: Tuple[int, int] = (12, 10),
                     view_angle: Tuple[int, int] = (30, -60),
                     output_path: Optional[str] = None) -> plt.Figure:
        """
        可视化3D DEM
        
        Args:
            figsize: 图形大小
            view_angle: 视角 (elev, azim)
            output_path: 输出路径
            
        Returns:
            fig: matplotlib图形对象
        """
        if self.elevation_map is None:
            self.generate_dem()
        
        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # 降采样以提高绘制性能
        step = 2
        X_sub = self.X[::step, ::step]
        Y_sub = self.Y[::step, ::step]
        elev_sub = self.elevation_map[::step, ::step]
        
        # 绘制表面
        surf = ax.plot_surface(X_sub, Y_sub, elev_sub, cmap='terrain',
                              alpha=0.9, linewidth=0, antialiased=True,
                              rstride=1, cstride=1, shade=True)
        
        # 等高线投影
        offset = self.elevation_map.min() - 200
        ax.contour(self.X, self.Y, self.elevation_map, zdir='z', offset=offset,
                  cmap='terrain', alpha=0.5, linewidths=0.5, levels=10)
        
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        ax.set_xlabel('东西距离 (km)')
        ax.set_ylabel('南北距离 (km)')
        ax.set_zlabel('高程 (m)')
        ax.set_title(f'{self.terrain_info["name"]} 3D地形')
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"已保存: {output_path}")
        
        return fig
    
    def save_dem(self, output_path: str):
        """保存DEM数据"""
        if self.elevation_map is None:
            self.generate_dem()
        
        np.savez(output_path,
                elevation=self.elevation_map,
                X=self.X,
                Y=self.Y,
                terrain_type=self.terrain_type,
                resolution_m=self.resolution_m,
                terrain_info=self.terrain_info)
        print(f"DEM数据已保存: {output_path}")
    
    def load_dem(self, input_path: str):
        """加载DEM数据"""
        data = np.load(input_path, allow_pickle=True)
        self.elevation_map = data['elevation']
        self.X = data['X']
        self.Y = data['Y']
        self.terrain_type = str(data['terrain_type'])
        self.resolution_m = float(data['resolution_m'])
        self.terrain_info = data['terrain_info'].item()
        print(f"DEM数据已加载: {input_path}")


def generate_figure_3_3(output_dir: str = "output/thesis_figures") -> Dict[str, Path]:
    """
    生成图3-3: 地形DEM可视化效果
    
    Returns:
        生成的文件路径字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # (a) 虹湾着陆区
    print("生成虹湾着陆区DEM...")
    iridum = HighFidelityTerrainModel('sinus_iridum')
    elev_iridum = iridum.generate_dem(size=500, seed=42)
    X_iridum, Y_iridum = iridum.X, iridum.Y
    
    im1 = axes[0].imshow(elev_iridum, extent=[0, 50, 0, 50],
                        cmap='terrain', origin='lower', aspect='equal')
    
    # 等高线
    levels1 = np.linspace(elev_iridum.min(), elev_iridum.max(), 12)
    axes[0].contour(X_iridum, Y_iridum, elev_iridum, levels=levels1,
                   colors='black', alpha=0.4, linewidths=0.5)
    
    axes[0].set_xlabel('东西距离 (km)', fontsize=11)
    axes[0].set_ylabel('南北距离 (km)', fontsize=11)
    axes[0].set_title('(a) 虹湾着陆区三维地形\n(高分辨率DEM)', fontsize=12, fontweight='bold')
    
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('高程 (m)', fontsize=10)
    
    # (b) 南极艾特肯盆地
    print("生成南极艾特肯盆地DEM...")
    spa = HighFidelityTerrainModel('spa_basin')
    elev_spa = spa.generate_dem(size=500, seed=123)
    X_spa, Y_spa = spa.X, spa.Y
    
    im2 = axes[1].imshow(elev_spa, extent=[0, 100, 0, 100],
                        cmap='gist_earth', origin='lower', aspect='equal')
    
    levels2 = np.linspace(elev_spa.min(), elev_spa.max(), 15)
    axes[1].contour(X_spa, Y_spa, elev_spa, levels=levels2,
                   colors='black', alpha=0.4, linewidths=0.5)
    
    axes[1].set_xlabel('东西距离 (km)', fontsize=11)
    axes[1].set_ylabel('南北距离 (km)', fontsize=11)
    axes[1].set_title('(b) 南极艾特肯盆地三维地形\n(复杂地形DEM)', fontsize=12, fontweight='bold')
    
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('高程 (m)', fontsize=10)
    
    fig.suptitle('图3-3 地形DEM可视化效果', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    for fmt in ['png', 'pdf', 'eps']:
        path = output_dir / f"fig_3_3_dem_visualization_hf.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        results[fmt] = path
        print(f"已保存: {path}")
    
    plt.close(fig)
    
    # 保存DEM数据
    iridum.save_dem(output_dir / "sinus_iridum_dem.npz")
    spa.save_dem(output_dir / "spa_basin_dem.npz")
    
    # 输出统计信息
    print("\n=== 地形统计信息 ===")
    print("\n虹湾着陆区:")
    for k, v in iridum.get_terrain_statistics().items():
        print(f"  {k}: {v:.2f}")
    
    print("\n南极艾特肯盆地:")
    for k, v in spa.get_terrain_statistics().items():
        print(f"  {k}: {v:.2f}")
    
    return results


if __name__ == "__main__":
    # 测试高保真建模
    print("=" * 60)
    print("高保真月球地形建模 - 图3-3生成")
    print("=" * 60)
    
    results = generate_figure_3_3()
    
    print("\n" + "=" * 60)
    print("完成!")
    print(f"输出文件: {results}")
    print("=" * 60)

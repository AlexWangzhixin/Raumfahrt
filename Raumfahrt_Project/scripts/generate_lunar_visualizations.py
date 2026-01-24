#!/usr/bin/env python3
"""
生成月球环境可视化图片
包括大尺度米级分辨率月面建模和厘米级语义可视化建模
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.modeling import EnvironmentModeling

def generate_large_scale_modeling():
    """
    生成大尺度米级分辨率月面建模图片
    """
    print("=== 生成大尺度米级分辨率月面建模图片 ===")
    
    # 初始化环境模型（大尺度）
    env_model = EnvironmentModeling(
        map_resolution=1.0,  # 米级分辨率
        map_size=(1000.0, 1000.0)  # 1000m × 1000m 大尺度区域
    )
    
    # 生成模拟地形数据
    generate_synthetic_terrain(env_model)
    
    # 生成语义分割数据
    semantic_segmentation = env_model.generate_random_semantic_segmentation()
    env_model._process_semantic_segmentation(semantic_segmentation)
    
    # 可视化大尺度月面模型
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 高程图
    im1 = ax1.imshow(env_model.elevation_map, cmap='terrain', origin='lower')
    ax1.set_title('大尺度月面高程图 (1m分辨率)')
    ax1.set_xlabel('X (像素)')
    ax1.set_ylabel('Y (像素)')
    cbar1 = fig.colorbar(im1, ax=ax1, label='高度 (m)')
    
    # 可通行性图
    im2 = ax2.imshow(env_model.traversability_map, cmap='viridis', origin='lower')
    ax2.set_title('大尺度月面可通行性图')
    ax2.set_xlabel('X (像素)')
    ax2.set_ylabel('Y (像素)')
    cbar2 = fig.colorbar(im2, ax=ax2, label='可通行性')
    
    # 物理属性图（凝聚力）
    im3 = ax3.imshow(env_model.physics_map[:, :, 0], cmap='plasma', origin='lower')
    ax3.set_title('大尺度月面物理属性图 (凝聚力)')
    ax3.set_xlabel('X (像素)')
    ax3.set_ylabel('Y (像素)')
    cbar3 = fig.colorbar(im3, ax=ax3, label='凝聚力 (Pa/m)')
    
    plt.tight_layout()
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存大尺度月面建模图片
    large_scale_path = os.path.join(output_dir, 'large_scale_lunar_modeling.png')
    plt.savefig(large_scale_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"大尺度月面建模图片已保存到: {large_scale_path}")
    return large_scale_path

def generate_cm_level_semantic_modeling():
    """
    生成厘米级语义可视化建模图片
    """
    print("\n=== 生成厘米级语义可视化建模图片 ===")
    
    # 初始化环境模型（厘米级）
    env_model = EnvironmentModeling(
        map_resolution=0.01,  # 厘米级分辨率
        map_size=(10.0, 10.0)  # 10m × 10m 小区域，厘米级细节
    )
    
    # 生成模拟地形数据（更精细）
    generate_detailed_terrain(env_model)
    
    # 生成语义分割数据（更精细）
    semantic_segmentation = generate_detailed_semantic_segmentation(env_model)
    env_model._process_semantic_segmentation(semantic_segmentation)
    
    # 可视化厘米级语义模型
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 高程图（厘米级细节）
    im1 = ax1.imshow(env_model.elevation_map, cmap='terrain', origin='lower')
    ax1.set_title('厘米级月面高程图 (1cm分辨率)')
    ax1.set_xlabel('X (像素)')
    ax1.set_ylabel('Y (像素)')
    cbar1 = fig.colorbar(im1, ax=ax1, label='高度 (m)')
    
    # 语义分割图
    im2 = ax2.imshow(semantic_segmentation, cmap='viridis', origin='lower')
    ax2.set_title('厘米级语义分割图')
    ax2.set_xlabel('X (像素)')
    ax2.set_ylabel('Y (像素)')
    cbar2 = fig.colorbar(im2, ax=ax2, label='语义标签')
    cbar2.set_ticks([0, 1, 2])
    cbar2.set_ticklabels(['松软月壤', '压实月壤', '岩石'])
    
    # 可通行性图
    im3 = ax3.imshow(env_model.traversability_map, cmap='plasma', origin='lower')
    ax3.set_title('厘米级月面可通行性图')
    ax3.set_xlabel('X (像素)')
    ax3.set_ylabel('Y (像素)')
    cbar3 = fig.colorbar(im3, ax=ax3, label='可通行性')
    
    # 物理属性图（摩擦角）
    im4 = ax4.imshow(env_model.physics_map[:, :, 4], cmap='magma', origin='lower', vmin=20, vmax=50)
    ax4.set_title('厘米级月面物理属性图 (摩擦角)')
    ax4.set_xlabel('X (像素)')
    ax4.set_ylabel('Y (像素)')
    cbar4 = fig.colorbar(im4, ax=ax4, label='摩擦角 (度)')
    
    plt.tight_layout()
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存厘米级语义可视化图片
    cm_level_path = os.path.join(output_dir, 'cm_level_semantic_modeling.png')
    plt.savefig(cm_level_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"厘米级语义可视化建模图片已保存到: {cm_level_path}")
    return cm_level_path

def generate_synthetic_terrain(env_model):
    """
    生成模拟地形数据（大尺度）
    """
    # 生成随机地形
    height, width = env_model.elevation_map.shape
    
    # 使用分形噪声生成更自然的地形
    env_model.elevation_map = generate_fractal_terrain(height, width, octaves=3, persistence=0.5)
    
    # 添加一些大尺度地形特征
    add_large_scale_features(env_model)
    
    # 归一化高程
    env_model.elevation_map = (env_model.elevation_map - np.min(env_model.elevation_map)) / \
                              (np.max(env_model.elevation_map) - np.min(env_model.elevation_map)) * 2 - 0.5
    
    # 基于地形生成大尺度可通性图
    generate_large_scale_traversability(env_model)

def generate_detailed_terrain(env_model):
    """
    生成详细的地形数据（厘米级）
    """
    # 生成更精细的地形
    height, width = env_model.elevation_map.shape
    
    # 使用分形噪声生成更自然的地形（无明显周期性）
    env_model.elevation_map = generate_fractal_terrain(height, width, octaves=4, persistence=0.6)
    
    # 添加一些小尺度地形特征
    add_small_scale_features(env_model)
    
    # 归一化高程
    env_model.elevation_map = (env_model.elevation_map - np.min(env_model.elevation_map)) / \
                              (np.max(env_model.elevation_map) - np.min(env_model.elevation_map)) * 0.5 - 0.1
    
    # 基于地形和语义分割生成厘米级可通性图
    generate_cm_level_traversability(env_model)

def generate_detailed_semantic_segmentation(env_model):
    """
    生成详细的语义分割数据（厘米级）
    """
    height, width = env_model.map_height, env_model.map_width
    semantic_segmentation = np.zeros((height, width), dtype=int)
    
    # 生成基础语义分布
    # 60% 压实月壤
    # 25% 松软月壤
    # 15% 岩石
    for i in range(height):
        for j in range(width):
            rand = np.random.rand()
            if rand < 0.15:
                semantic_segmentation[i, j] = 2  # 岩石
            elif rand < 0.40:
                semantic_segmentation[i, j] = 0  # 松软月壤
            else:
                semantic_segmentation[i, j] = 1  # 压实月壤
    
    # 添加一些连续的岩石区域
    for _ in range(5):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(10, 30)
        
        for i in range(max(0, center_y-radius), min(height, center_y+radius)):
            for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                if distance < radius:
                    semantic_segmentation[i, j] = 2  # 岩石
    
    # 添加一些连续的松软月壤区域
    for _ in range(8):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(15, 40)
        
        for i in range(max(0, center_y-radius), min(height, center_y+radius)):
            for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                if distance < radius:
                    semantic_segmentation[i, j] = 0  # 松软月壤
    
    return semantic_segmentation

def main():
    """
    主函数
    """
    print("开始生成月球环境可视化图片...")
    
    # 生成大尺度月面建模图片
    large_scale_path = generate_large_scale_modeling()
    
    # 生成厘米级语义可视化建模图片
    cm_level_path = generate_cm_level_semantic_modeling()
    
    print("\n=== 生成完成 ===")
    print(f"大尺度米级分辨率月面建模图片路径: {large_scale_path}")
    print(f"厘米级语义可视化建模图片路径: {cm_level_path}")
    
    return large_scale_path, cm_level_path

def generate_fractal_terrain(height, width, octaves=3, persistence=0.5):
    """
    生成分形噪声地形，避免周期性
    
    Args:
        height: 高度
        width: 宽度
        octaves: 八度数量
        persistence: 持久性
        
    Returns:
        分形地形数据
    """
    from scipy.ndimage import gaussian_filter
    
    # 初始化地形
    terrain = np.zeros((height, width))
    
    # 分形噪声生成
    for octave in range(octaves):
        scale = 2 ** octave
        noise = np.random.rand(height // scale + 1, width // scale + 1)
        
        # 上采样
        from skimage.transform import resize
        noise_up = resize(noise, (height, width), order=1)
        
        # 添加到地形
        terrain += noise_up * (persistence ** octave)
    
    # 平滑处理
    terrain = gaussian_filter(terrain, sigma=1)
    
    return terrain

def add_large_scale_features(env_model):
    """
    添加大尺度地形特征
    """
    height, width = env_model.elevation_map.shape
    
    # 添加一些山丘
    for _ in range(5):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(50, 150)
        
        for i in range(max(0, center_y-radius), min(height, center_y+radius)):
            for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                if distance < radius:
                    env_model.elevation_map[i, j] += 0.5 * (1 - distance/radius)
    
    # 添加一些洼地
    for _ in range(3):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(30, 100)
        
        for i in range(max(0, center_y-radius), min(height, center_y+radius)):
            for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                if distance < radius:
                    env_model.elevation_map[i, j] -= 0.3 * (1 - distance/radius)

def add_small_scale_features(env_model):
    """
    添加小尺度地形特征
    """
    height, width = env_model.elevation_map.shape
    
    # 添加一些小岩石
    for _ in range(20):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(2, 8)
        
        for i in range(max(0, center_y-radius), min(height, center_y+radius)):
            for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                if distance < radius:
                    env_model.elevation_map[i, j] += 0.05 * (1 - distance/radius)
    
    # 添加一些小洼地
    for _ in range(10):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(3, 10)
        
        for i in range(max(0, center_y-radius), min(height, center_y+radius)):
            for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                distance = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                if distance < radius:
                    env_model.elevation_map[i, j] -= 0.03 * (1 - distance/radius)

def generate_large_scale_traversability(env_model):
    """
    生成大尺度可通性图
    """
    height, width = env_model.elevation_map.shape
    
    # 基于高程计算坡度
    from scipy.ndimage import sobel
    dx = sobel(env_model.elevation_map, axis=1)
    dy = sobel(env_model.elevation_map, axis=0)
    gradient = np.sqrt(dx**2 + dy**2)
    
    # 坡度越大，可通性越低
    slope_factor = np.exp(-gradient * 5)
    
    # 基于高程的可通性（过高或过低的区域可通性低）
    elevation_factor = np.exp(-(env_model.elevation_map)**2 / 0.5)
    
    # 综合可通性
    env_model.traversability_map = 0.6 * slope_factor + 0.4 * elevation_factor
    
    # 确保可通性在0-1之间
    env_model.traversability_map = np.clip(env_model.traversability_map, 0, 1)

def generate_cm_level_traversability(env_model):
    """
    生成厘米级可通性图
    """
    height, width = env_model.elevation_map.shape
    
    # 基于高程计算坡度
    from scipy.ndimage import sobel
    dx = sobel(env_model.elevation_map, axis=1)
    dy = sobel(env_model.elevation_map, axis=0)
    gradient = np.sqrt(dx**2 + dy**2)
    
    # 坡度越大，可通性越低
    slope_factor = np.exp(-gradient * 10)
    
    # 基于高程的可通性
    elevation_factor = np.exp(-(env_model.elevation_map)**2 / 0.2)
    
    # 基于物理属性的可通性
    # 从物理属性地图获取信息
    cohesion = env_model.physics_map[:, :, 0]
    friction_angle = env_model.physics_map[:, :, 4]
    
    # 凝聚力越大，可通性越高
    cohesion_factor = np.exp(-cohesion / 1e4)
    cohesion_factor = np.clip(cohesion_factor, 0, 1)
    
    # 摩擦角越大，可通性越高
    friction_factor = friction_angle / 50.0
    friction_factor = np.clip(friction_factor, 0, 1)
    
    # 综合可通性
    env_model.traversability_map = 0.3 * slope_factor + 0.2 * elevation_factor + 0.3 * cohesion_factor + 0.2 * friction_factor
    
    # 确保可通性在0-1之间
    env_model.traversability_map = np.clip(env_model.traversability_map, 0, 1)

if __name__ == "__main__":
    main()
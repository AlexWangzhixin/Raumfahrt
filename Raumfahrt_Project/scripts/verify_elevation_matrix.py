#!/usr/bin/env python3
"""
验证和演示如何使用NAC_DTM_CHANGE4.tiff的原始高程矩阵
"""

import os
import numpy as np

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 高程矩阵文件路径
elevation_matrix_path = os.path.join(project_root, 'outputs', 'visualizations', 'tiff', 'nac_dtm_elevation_matrix.npy')

def load_elevation_matrix():
    """
    加载原始高程矩阵
    
    Returns:
        numpy.ndarray: 原始高程矩阵
    """
    print(f"加载高程矩阵: {elevation_matrix_path}")
    
    try:
        # 加载.npy文件
        elevation_matrix = np.load(elevation_matrix_path)
        
        print(f"高程矩阵加载成功！")
        print(f"矩阵形状: {elevation_matrix.shape}")
        print(f"矩阵类型: {elevation_matrix.dtype}")
        
        # 计算基本统计信息（跳过无效值）
        valid_mask = (elevation_matrix != -3.4028226550889045e+38) & (~np.isinf(elevation_matrix)) & (~np.isnan(elevation_matrix))
        valid_data = elevation_matrix[valid_mask]
        
        if len(valid_data) > 0:
            print(f"\n有效数据统计:")
            print(f"  有效数据点数量: {len(valid_data)}")
            print(f"  高程范围: {np.min(valid_data):.2f} - {np.max(valid_data):.2f}")
            print(f"  高程均值: {np.mean(valid_data):.2f}")
            print(f"  高程标准差: {np.std(valid_data):.2f}")
        else:
            print("警告: 没有找到有效数据")
        
        return elevation_matrix
    except Exception as e:
        print(f"加载高程矩阵失败: {e}")
        return None

def demonstrate_usage(elevation_matrix):
    """
    演示如何使用高程矩阵
    
    Args:
        elevation_matrix: 原始高程矩阵
    """
    if elevation_matrix is None:
        return
    
    print("\n=== 高程矩阵使用示例 ===")
    
    # 示例1: 获取特定位置的高程值
    print("示例1: 获取特定位置的高程值")
    x, y = 100, 100
    elevation_value = elevation_matrix[y, x]
    print(f"  位置 ({x}, {y}) 的高程值: {elevation_value:.2f}")
    
    # 示例2: 裁剪子区域
    print("\n示例2: 裁剪子区域")
    sub_matrix = elevation_matrix[500:1000, 500:1000]
    print(f"  子区域形状: {sub_matrix.shape}")
    
    # 示例3: 计算坡度（简单实现）
    print("\n示例3: 计算坡度（简单实现）")
    try:
        # 计算梯度
        dy, dx = np.gradient(elevation_matrix[:100, :100])  # 只计算一小部分以提高速度
        slope = np.sqrt(dx**2 + dy**2)
        print(f"  坡度范围: {np.min(slope):.2f} - {np.max(slope):.2f}")
        print(f"  坡度均值: {np.mean(slope):.2f}")
    except Exception as e:
        print(f"  计算坡度失败: {e}")

def main():
    """
    主函数
    """
    print("=== NAC_DTM_CHANGE4 高程矩阵验证 ===")
    
    # 加载高程矩阵
    elevation_matrix = load_elevation_matrix()
    
    if elevation_matrix is not None:
        # 演示使用方法
        demonstrate_usage(elevation_matrix)
    
    print("\n=== 验证完成 ===")

if __name__ == "__main__":
    main()

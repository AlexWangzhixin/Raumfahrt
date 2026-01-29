#!/usr/bin/env python3
"""
可视化TIFF文件（NAC_DTM_CHANGE4.tiff）
参考嫦娥6号着陆区图像的风格
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TiffVisualizer:
    """
    TIFF文件可视化器
    """
    
    def __init__(self, tiff_path, reference_dir=None):
        """
        初始化TIFF可视化器
        
        Args:
            tiff_path: TIFF文件路径
            reference_dir: 参考图像目录（嫦娥6号着陆区图像）
        """
        self.tiff_path = tiff_path
        self.reference_dir = reference_dir
        self.data = None
        self.metadata = {}
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_tiff(self):
        """
        加载TIFF文件
        """
        print(f"加载TIFF文件: {self.tiff_path}")
        
        try:
            # 使用PIL加载TIFF文件
            img = Image.open(self.tiff_path)
            self.data = np.array(img)
            
            # 数据清洗和验证
            self._clean_data()
            
            # 获取基本信息
            self.metadata = {
                'shape': self.data.shape,
                'min_value': np.min(self.data),
                'max_value': np.max(self.data),
                'mean_value': np.mean(self.data),
                'std_value': np.std(self.data),
                'original_min': np.min(self.original_data),
                'original_max': np.max(self.original_data),
                'original_mean': np.mean(self.original_data),
                'original_std': np.std(self.original_data)
            }
            
            print(f"TIFF文件信息:")
            print(f"  形状: {self.metadata['shape']}")
            print(f"  归一化值范围: {self.metadata['min_value']:.2f} - {self.metadata['max_value']:.2f}")
            print(f"  归一化均值: {self.metadata['mean_value']:.2f}")
            print(f"  归一化标准差: {self.metadata['std_value']:.2f}")
            print(f"  原始高程范围: {self.metadata['original_min']:.2f} - {self.metadata['original_max']:.2f}")
            print(f"  原始高程均值: {self.metadata['original_mean']:.2f}")
            print(f"  原始高程标准差: {self.metadata['original_std']:.2f}")
            
            return True
        except Exception as e:
            print(f"加载TIFF文件失败: {e}")
            return False
    
    def _clean_data(self):
        """
        清洗和预处理数据
        """
        print("清洗数据...")
        
        # 保存原始数据（未归一化）
        self.original_data = self.data.copy()
        
        # 替换无效值
        invalid_mask = (self.data == -3.4028226550889045e+38) | (np.isinf(self.data)) | (np.isnan(self.data))
        
        if np.any(invalid_mask):
            print(f"发现 {np.sum(invalid_mask)} 个无效值，正在替换...")
            # 使用有效数据的平均值替换无效值
            valid_data = self.data[~invalid_mask]
            if len(valid_data) > 0:
                mean_valid = np.mean(valid_data)
                self.data[invalid_mask] = mean_valid
                print(f"使用有效值的平均值 {mean_valid:.2f} 替换无效值")
            else:
                print("警告: 没有有效值，使用0替换")
                self.data[invalid_mask] = 0
        
        # 确保数据类型正确
        self.data = self.data.astype(np.float64)
        
        # 调整数据范围（如果需要）
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        if max_val > min_val:
            # 归一化到合适范围
            self.data = (self.data - min_val) / (max_val - min_val) * 255
            print("数据已归一化到0-255范围")
    
    def analyze_reference_style(self):
        """
        分析参考图像的风格
        """
        if not self.reference_dir:
            print("未提供参考图像目录，使用默认风格参数")
            return self.get_default_style()
        
        print(f"分析参考图像风格: {self.reference_dir}")
        
        # 这里可以添加更复杂的风格分析
        # 目前使用基于嫦娥6号着陆区图像的默认参数
        style_params = {
            'cmap': 'terrain',  # 地形配色方案
            'vmin': self.metadata.get('min_value', 0),
            'vmax': self.metadata.get('max_value', 255),
            'title': 'NAC_DTM_CHANGE4 地形可视化',
            'figsize': (12, 10),
            'dpi': 150,
            'colorbar_label': '高程 (m)',
            'extent': None  # 可以根据实际地理坐标设置
        }
        
        return style_params
    
    def get_default_style(self):
        """
        获取默认风格参数
        """
        return {
            'cmap': 'terrain',
            'vmin': self.metadata.get('min_value', 0),
            'vmax': self.metadata.get('max_value', 255),
            'title': 'NAC_DTM_CHANGE4 地形可视化',
            'figsize': (12, 10),
            'dpi': 150,
            'colorbar_label': '高程 (m)',
            'extent': None
        }
    
    def visualize(self, output_path=None):
        """
        可视化TIFF文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.data is None:
            print("错误: 请先加载TIFF文件")
            return False
        
        # 分析风格参数
        style_params = self.analyze_reference_style()
        
        print("创建可视化...")
        
        # 创建图表
        plt.figure(figsize=style_params['figsize'], dpi=style_params['dpi'])
        
        # 绘制数据
        im = plt.imshow(
            self.data,
            cmap=style_params['cmap'],
            vmin=style_params['vmin'],
            vmax=style_params['vmax'],
            extent=style_params['extent']
        )
        
        # 添加标题和颜色条
        plt.title(style_params['title'])
        cbar = plt.colorbar(im, label=style_params['colorbar_label'])
        
        # 设置轴标签
        plt.xlabel('X 像素')
        plt.ylabel('Y 像素')
        
        # 添加网格（可选）
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if output_path:
            # 确保使用正确的路径分隔符
            output_path = os.path.normpath(output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"创建目录: {output_dir}")
            
            plt.savefig(output_path, dpi=style_params['dpi'], bbox_inches='tight')
            print(f"可视化结果保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def visualize_with_analysis(self, output_dir=None):
        """
        可视化TIFF文件并添加分析信息
        
        Args:
            output_dir: 输出目录
        """
        if self.data is None:
            print("错误: 请先加载TIFF文件")
            return False
        
        if not output_dir:
            output_dir = 'outputs/visualizations'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 主可视化
        main_output = os.path.join(output_dir, 'nac_dtm_visualization.png')
        self.visualize(main_output)
        
        # 2. 直方图分析
        hist_output = os.path.join(output_dir, 'nac_dtm_histogram.png')
        self.plot_histogram(hist_output)
        
        # 3. 统计信息文件
        stats_output = os.path.join(output_dir, 'nac_dtm_stats.txt')
        self.save_statistics(stats_output)
        
        # 4. 保存原始高程矩阵
        elevation_output = os.path.join(output_dir, 'nac_dtm_elevation_matrix.npy')
        self.save_elevation_matrix(elevation_output)
        
        return True
    
    def plot_histogram(self, output_path):
        """
        绘制数据直方图
        """
        plt.figure(figsize=(10, 6), dpi=150)
        plt.hist(self.data.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('NAC_DTM_CHANGE4 数据分布')
        plt.xlabel('值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"直方图保存到: {output_path}")
    
    def get_elevation_matrix(self):
        """
        获取原始高程矩阵
        
        Returns:
            numpy.ndarray: 原始高程矩阵（未归一化）
        """
        if hasattr(self, 'original_data'):
            return self.original_data
        return self.data
    
    def save_statistics(self, output_path):
        """
        保存统计信息
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('NAC_DTM_CHANGE4 统计信息\n')
            f.write('=' * 40 + '\n')
            
            # 基本信息
            f.write('基本信息:\n')
            f.write(f'  形状: {self.metadata["shape"]}\n')
            
            # 归一化数据信息
            f.write('\n归一化数据（用于可视化）:\n')
            f.write(f'  值范围: {self.metadata["min_value"]:.2f} - {self.metadata["max_value"]:.2f}\n')
            f.write(f'  均值: {self.metadata["mean_value"]:.2f}\n')
            f.write(f'  标准差: {self.metadata["std_value"]:.2f}\n')
            
            # 原始高程数据信息
            f.write('\n原始高程数据:\n')
            f.write(f'  高程范围: {self.metadata["original_min"]:.2f} - {self.metadata["original_max"]:.2f}\n')
            f.write(f'  高程均值: {self.metadata["original_mean"]:.2f}\n')
            f.write(f'  高程标准差: {self.metadata["original_std"]:.2f}\n')
            
            f.write('=' * 40 + '\n')
            f.write('可视化参数:\n')
            f.write(f'  配色方案: terrain\n')
            f.write(f'  参考风格: 嫦娥6号着陆区图像\n')
        print(f"统计信息保存到: {output_path}")
    
    def save_elevation_matrix(self, output_path):
        """
        保存原始高程矩阵
        
        Args:
            output_path: 输出文件路径（.npy格式）
        """
        elevation_matrix = self.get_elevation_matrix()
        np.save(output_path, elevation_matrix)
        print(f"原始高程矩阵保存到: {output_path}")
        return True

def main():
    """
    主函数
    """
    print("=== NAC_DTM_CHANGE4.tiff 可视化 ===")
    
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # TIFF文件路径
    tiff_path = os.path.join(project_root, '..', 'data', 'dtm', 'NAC_DTM_CHANGE4.tiff')
    
    # 参考图像目录（嫦娥6号着陆区）
    reference_dir = os.path.join(project_root, '..', 'data', '嫦娥6着陆区')
    
    # 输出目录
    output_dir = os.path.join(project_root, 'outputs', 'visualizations', 'tiff')
    
    # 打印路径信息
    print(f"项目根目录: {project_root}")
    print(f"TIFF文件路径: {tiff_path}")
    print(f"参考图像目录: {reference_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(tiff_path):
        print(f"错误: TIFF文件不存在: {tiff_path}")
        return
    
    if not os.path.exists(reference_dir):
        print(f"警告: 参考图像目录不存在: {reference_dir}")
    
    # 创建可视化器
    visualizer = TiffVisualizer(tiff_path, reference_dir)
    
    # 加载TIFF文件
    if not visualizer.load_tiff():
        print("加载TIFF文件失败，退出")
        return
    
    # 执行可视化和分析
    visualizer.visualize_with_analysis(output_dir)
    
    print("\n=== 可视化完成 ===")

if __name__ == "__main__":
    main()

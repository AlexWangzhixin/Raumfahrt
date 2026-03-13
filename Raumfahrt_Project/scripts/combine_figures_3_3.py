#!/usr/bin/env python3
"""
将图3-3的2D和3D版本上下拼接
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

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

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "thesis_figures"
INPUT_DIR = OUTPUT_DIR


def combine_figures():
    """上下拼接2D和3D图"""
    
    # 读取2D和3D图像
    img_2d_path = INPUT_DIR / "fig_3_3_dem_visualization.png"
    img_3d_path = INPUT_DIR / "fig_3_3_dem_3d_visualization.png"
    
    if not img_2d_path.exists():
        print(f"错误: 找不到2D图像 {img_2d_path}")
        return
    if not img_3d_path.exists():
        print(f"错误: 找不到3D图像 {img_3d_path}")
        return
    
    img_2d = Image.open(img_2d_path)
    img_3d = Image.open(img_3d_path)
    
    print(f"2D图像尺寸: {img_2d.size}")
    print(f"3D图像尺寸: {img_3d.size}")
    
    # 调整宽度一致
    target_width = 2400
    
    # 计算新高度保持比例
    ratio_2d = target_width / img_2d.width
    new_height_2d = int(img_2d.height * ratio_2d)
    
    ratio_3d = target_width / img_3d.width
    new_height_3d = int(img_3d.height * ratio_3d)
    
    # 调整大小
    img_2d_resized = img_2d.resize((target_width, new_height_2d), Image.LANCZOS)
    img_3d_resized = img_3d.resize((target_width, new_height_3d), Image.LANCZOS)
    
    # 创建新图像（上下拼接）
    total_height = new_height_2d + new_height_3d
    combined = Image.new('RGB', (target_width, total_height), 'white')
    
    # 粘贴图像
    combined.paste(img_2d_resized, (0, 0))
    combined.paste(img_3d_resized, (0, new_height_2d))
    
    # 保存
    output_path = OUTPUT_DIR / "fig_3_3_combined.png"
    combined.save(output_path, dpi=(300, 300))
    print(f"已保存拼接图像: {output_path}")
    
    # 同时保存为PDF
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), dpi=150)
    axes[0].imshow(img_2d_resized)
    axes[0].axis('off')
    axes[0].set_title('(a) 平面视角 DEM', fontsize=14, fontweight='bold')
    
    axes[1].imshow(img_3d_resized)
    axes[1].axis('off')
    axes[1].set_title('(b) 三维视角 DEM', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    pdf_path = OUTPUT_DIR / "fig_3_3_combined.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存PDF: {pdf_path}")
    
    plt.close()
    
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("拼接图3-3的2D和3D版本")
    print("=" * 60)
    
    configure_chinese_font()
    combine_figures()
    
    print("=" * 60)
    print("完成!")
    print("=" * 60)

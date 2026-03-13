#!/usr/bin/env python3
"""
分析图3-3中的字体大小，与正文12pt字体对比
"""

from PIL import Image
import numpy as np
from pathlib import Path

# 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = PROJECT_ROOT / "output" / "thesis_figures" / "fig_3_3_final_combined.png"

def analyze_font_size():
    """分析图片中字体大小"""
    
    if not IMAGE_PATH.exists():
        print(f"错误: 找不到图片 {IMAGE_PATH}")
        return
    
    img = Image.open(IMAGE_PATH)
    width, height = img.size
    
    print("=" * 60)
    print("图3-3字体大小分析")
    print("=" * 60)
    print(f"\n图片尺寸: {width} x {height} 像素")
    
    # 假设图片在A4纸中占0.95\textwidth
    # A4纸宽度 = 210mm
    # 正文宽度 = 0.95 * 210mm ≈ 200mm
    text_width_mm = 0.95 * 210
    
    # 像素到mm的转换
    px_per_mm = width / text_width_mm
    print(f"图片在A4中的宽度: {text_width_mm:.1f} mm")
    print(f"像素密度: {px_per_mm:.1f} px/mm")
    
    # 正文字体 = 小四号 = 12pt
    # 1pt = 0.3528mm
    font_size_pt = 12
    font_size_mm = font_size_pt * 0.3528
    font_size_px = font_size_mm * px_per_mm
    
    print(f"\n正文字体: 小四号 = {font_size_pt}pt = {font_size_mm:.2f}mm = {font_size_px:.1f}px")
    
    # 中文字符实际高度约为字体大小的1.2-1.3倍（考虑行距）
    chinese_char_height_px = font_size_px * 1.2
    print(f"中文字符估算高度: {chinese_char_height_px:.1f}px")
    
    # 观察图片中文字高度
    # 从图片观察，标题文字高度约50-60像素，坐标轴文字约35-45像素
    observed_title_height = 55  # 估计值
    observed_axis_height = 40   # 估计值
    
    print(f"\n图片中观察到的文字高度:")
    print(f"  标题文字: ~{observed_title_height}px")
    print(f"  坐标轴文字: ~{observed_axis_height}px")
    
    # 计算比例
    title_ratio = observed_title_height / chinese_char_height_px
    axis_ratio = observed_axis_height / chinese_char_height_px
    
    print(f"\n与正文字体对比:")
    print(f"  标题文字 / 正文: {title_ratio:.2f} ({title_ratio*100:.0f}%)")
    print(f"  坐标轴文字 / 正文: {axis_ratio:.2f} ({axis_ratio*100:.0f}%)")
    
    # 判断是否满足要求（误差不超过10%）
    tolerance = 0.10
    title_ok = abs(title_ratio - 1.0) <= tolerance
    axis_ok = abs(axis_ratio - 1.0) <= tolerance
    
    print(f"\n判断结果 (允许误差 ±{tolerance*100:.0f}%):")
    print(f"  标题文字: {'✓ 符合要求' if title_ok else '✗ 需要调整'}")
    print(f"  坐标轴文字: {'✓ 符合要求' if axis_ok else '✗ 需要调整'}")
    
    if not (title_ok and axis_ok):
        # 计算需要的缩放比例
        recommended_scale = chinese_char_height_px / observed_axis_height
        print(f"\n建议调整:")
        print(f"  当前字体大小需要缩放: {recommended_scale:.2f}x")
        print(f"  建议matplotlib字体大小: {10 * recommended_scale:.1f}pt")
    
    print("\n" + "=" * 60)
    
    return {
        'font_size_px': chinese_char_height_px,
        'observed_axis_height': observed_axis_height,
        'ratio': axis_ratio,
        'needs_adjustment': not axis_ok
    }

if __name__ == "__main__":
    analyze_font_size()

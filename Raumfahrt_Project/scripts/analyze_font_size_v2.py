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
        print(f"Error: Image not found {IMAGE_PATH}")
        return
    
    img = Image.open(IMAGE_PATH)
    width, height = img.size
    
    print("=" * 60)
    print("Figure 3-3 Font Size Analysis")
    print("=" * 60)
    print(f"\nImage size: {width} x {height} pixels")
    
    # A4 paper width = 210mm
    # Text width = 0.95 * 210mm ≈ 200mm
    text_width_mm = 0.95 * 210
    
    # Convert pixels to mm
    px_per_mm = width / text_width_mm
    mm_per_px = text_width_mm / width
    
    print(f"Text width in A4: {text_width_mm:.1f} mm")
    print(f"Pixel density: {px_per_mm:.1f} px/mm")
    print(f"mm per pixel: {mm_per_px:.3f} mm/px")
    
    # Body text = 小四号 = 12pt
    # 1pt = 0.3528mm
    font_size_pt = 12
    font_size_mm = font_size_pt * 0.3528
    font_size_px = font_size_mm * px_per_mm
    
    print(f"\nBody text (small 4): {font_size_pt}pt = {font_size_mm:.2f}mm = {font_size_px:.1f}px")
    
    # Chinese character height is about 1.2x font size
    chinese_char_height_px = font_size_px * 1.2
    print(f"Estimated Chinese char height: {chinese_char_height_px:.1f}px")
    
    # Observed font heights from image (estimated)
    observed_title_height = 65   # Title text in image
    observed_axis_height = 50    # Axis label text in image
    observed_tick_height = 45    # Tick label text in image
    
    print(f"\nObserved text heights in image:")
    print(f"  Title text: ~{observed_title_height}px")
    print(f"  Axis labels: ~{observed_axis_height}px")
    print(f"  Tick labels: ~{observed_tick_height}px")
    
    # Calculate ratios
    title_ratio = observed_title_height / chinese_char_height_px
    axis_ratio = observed_axis_height / chinese_char_height_px
    tick_ratio = observed_tick_height / chinese_char_height_px
    
    print(f"\nComparison with body text:")
    print(f"  Title / Body: {title_ratio:.2f} ({title_ratio*100:.0f}%)")
    print(f"  Axis / Body: {axis_ratio:.2f} ({axis_ratio*100:.0f}%)")
    print(f"  Tick / Body: {tick_ratio:.2f} ({tick_ratio*100:.0f}%)")
    
    # Check if within tolerance (10%)
    tolerance = 0.10
    title_ok = abs(title_ratio - 1.0) <= tolerance
    axis_ok = abs(axis_ratio - 1.0) <= tolerance
    tick_ok = abs(tick_ratio - 1.0) <= tolerance
    
    print(f"\nResult (tolerance ±{tolerance*100:.0f}%):")
    print(f"  Title text: {'PASS' if title_ok else 'NEEDS ADJUSTMENT'}")
    print(f"  Axis labels: {'PASS' if axis_ok else 'NEEDS ADJUSTMENT'}")
    print(f"  Tick labels: {'PASS' if tick_ok else 'NEEDS ADJUSTMENT'}")
    
    all_pass = title_ok and axis_ok and tick_ok
    
    if not all_pass:
        # Calculate recommended scale
        target_ratio = 1.0
        if not axis_ok:
            recommended_scale = target_ratio / axis_ratio
            print(f"\nRecommendation:")
            print(f"  Current axis font ratio: {axis_ratio:.2f}")
            print(f"  Recommended scale factor: {recommended_scale:.2f}x")
            print(f"  Current matplotlib font size: 13pt")
            print(f"  Recommended font size: {13 * recommended_scale:.1f}pt")
    else:
        print(f"\n[OK] All font sizes are within ±{tolerance*100:.0f}% of body text!")
    
    print("\n" + "=" * 60)
    
    return {
        'font_size_px': chinese_char_height_px,
        'observed_axis_height': observed_axis_height,
        'axis_ratio': axis_ratio,
        'all_pass': all_pass
    }

if __name__ == "__main__":
    result = analyze_font_size()
    
    # Exit with error code if adjustment needed
    if result and not result['all_pass']:
        exit(1)

# UML 架构图绘制 Skill

本 Skill 提供专业的 UML 2.0 风格架构图绘制能力，特别适用于学术论文中的系统架构图、数字孪生架构图等复杂技术图表。

## 能力概述

- 绘制 UML 包图、组件图、部署图
- 支持中文优化显示
- 自动处理中英文混排
- 避免元素重叠
- 支持多种布局方向（横向/竖向）

## 使用规范

### 1. 文本规范

**英文处理原则：**
- 尽量使用中文表达
- 必要英文缩写需在首次出现时括号说明
- 常见缩写对照表：
  - API → 应用程序接口（API）
  - SLAM → 同步定位与地图构建（SLAM）
  - KF/PF → 卡曼滤波/粒子滤波
  - DEM → 数字高程模型（DEM）
  - UML → 统一建模语言（UML）

### 2. 布局规范

**竖排布局（推荐）：**
- 适用于层级分明的架构图
- 画布比例：宽度 5.8，高度 11.69（A4竖版）
- 每层之间预留 0.3-0.5 空间用于连接线

**横向布局：**
- 适用于并行模块展示
- 画布比例：宽度 11.69，高度 8.27（A4横版）

### 3. 防重叠规则

**文字与边框：**
- 组件标题与顶部边框距离 ≥ 0.18
- 子项与分隔线距离 ≥ 0.22
- 行间距根据内容自动计算

**连接线标注：**
- 使用白色背景文本框
- 标注与线条距离 ≥ 0.1
- 优先使用箭头连接而非直线

### 4. 颜色规范

| 元素类型 | 背景色 | 边框色 | 文字色 |
|---------|--------|--------|--------|
| 包（Package） | #f5f5f5 | #424242 | 白色（标题） |
| 组件（Component） | #e8f4f8 | #1565c0 | #424242 |
| 数据层 | #f3e5f5 | #7b1fa2 | #424242 |
| 服务层 | #e8f5e9 | #388e3c | #424242 |
| 注释 | #fff8e1 | #f9a825 | #424242 |

## 示例代码

```python
#!/usr/bin/env python3
"""UML架构图绘制示例"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

# 必须设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布 - 竖版A4，横向缩短
def create_figure(width=5.8, height=11.69, dpi=200):
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    return fig, ax

# 绘制UML包
def draw_package(ax, x, y, width, height, name, color_key, border_color):
    """绘制UML包"""
    tab_width = min(len(name) * 0.22 + 0.3, width * 0.7)
    tab_height = 0.35
    
    # 包标签
    tab = FancyBboxPatch((x, y + height - tab_height), tab_width, tab_height,
                         boxstyle="round,pad=0,rounding_size=0.04",
                         facecolor=border_color,
                         edgecolor='#424242',
                         linewidth=1.2)
    ax.add_patch(tab)
    ax.text(x + tab_width/2, y + height - tab_height/2, name, 
            ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # 包体
    body = FancyBboxPatch((x, y), width, height - tab_height,
                          boxstyle="round,pad=0,rounding_size=0.04",
                          facecolor=color_key,
                          edgecolor='#424242',
                          linewidth=1.2)
    ax.add_patch(body)

# 绘制UML组件
def draw_component(ax, x, y, width, height, name, sub_items, 
                   bg_color, border_color):
    """绘制UML组件 - 自动防重叠"""
    # 组件体
    comp = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0,rounding_size=0.03",
                          facecolor=bg_color,
                          edgecolor=border_color,
                          linewidth=1.2)
    ax.add_patch(comp)
    
    # 组件图标
    icon_x = x + 0.06
    for i, icon_y in enumerate([y + height - 0.28, y + height - 0.4]):
        rect = Rectangle((icon_x, icon_y), 0.08, 0.08, 
                        facecolor='white', edgecolor=border_color, linewidth=0.8)
        ax.add_patch(rect)
    
    # 标题
    ax.text(x + width/2 + 0.05, y + height - 0.18, name, 
            ha='center', va='center',
            fontsize=8, fontweight='bold', color=border_color)
    
    # 分隔线
    line_y = y + height - 0.5
    ax.plot([x + 0.04, x + width - 0.04], [line_y, line_y], 
            color=border_color, linewidth=0.6)
    
    # 子项 - 自动计算行距
    item_y_start = line_y - 0.22
    line_height = (item_y_start - y - 0.15) / max(len(sub_items), 1)
    
    for i, item in enumerate(sub_items):
        item_y = item_y_start - i * line_height
        fontsize = 6.5 if len(item) > 18 else 7
        ax.text(x + 0.18, item_y, item, ha='left', va='center',
                fontsize=fontsize, color='#424242')

# 绘制带背景的文字（防重叠）
def draw_text_with_bg(ax, x, y, text, fontsize=7, color='#424242'):
    """绘制带白色背景的文字，避免与线条重叠"""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=color,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                     edgecolor='none', alpha=0.9))

# 绘制连接线
def draw_connector(ax, x1, y1, x2, y2, label='', style='-'):
    """绘制带标注的连接线"""
    if style == '--':
        ax.plot([x1, x2], [y1, y2], color='#666', linewidth=0.8, linestyle='--')
    else:
        ax.plot([x1, x2], [y1, y2], color='#333', linewidth=1)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        draw_text_with_bg(ax, mid_x, mid_y + 0.1, label, fontsize=6)
```

## 完整示例

参见项目中的 `create_uml_architecture_v4.py` 文件，这是一个完整的数字孪生架构图实现。

## 更新日志

### v1.0 (2026-03-10)
- 初始版本
- 支持竖排布局，横向缩短2/3
- 实现自动防重叠机制
- 建立中英文混排规范

## 依赖

- matplotlib >= 3.9
- numpy >= 2.0

## 作者

Raumfahrt Project

#!/usr/bin/env python3
"""
UML 2.0风格的五维数字孪生架构图 - v13最终版
- 下移服务层组件，与层标题保持充足间距
- 整体美感优化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import matplotlib.patheffects as path_effects

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布 - 调整尺寸以获得更好的比例
fig, ax = plt.subplots(figsize=(8.27, 10.2), dpi=200)
ax.set_xlim(0, 8.27)
ax.set_ylim(0, 10.2)
ax.axis('off')

# 颜色定义 - 微调使更协调
colors = {
    'package': '#f5f5f5',
    'package_border': '#333333',
    'component': '#e3f2fd',
    'component_border': '#1565c0',
    'data': '#f3e5f5',
    'data_border': '#6a1b9a',
    'service': '#e8f5e9',
    'service_border': '#2e7d32',
    'note': '#fff9c4',
    'note_border': '#f57f17',
    'title': '#212121',
    'text': '#333333'
}

def draw_package(ax, x, y, width, height, name, color_key, border_color):
    """绘制UML包"""
    tab_width = min(len(name) * 0.28 + 0.3, width * 0.85)
    tab = FancyBboxPatch((x, y + height - 0.42), tab_width, 0.42,
                         boxstyle="round,pad=0,rounding_size=0.05",
                         facecolor=border_color,
                         edgecolor=colors['package_border'],
                         linewidth=1.5)
    ax.add_patch(tab)
    ax.text(x + tab_width/2, y + height - 0.21, name, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    
    body = FancyBboxPatch((x, y), width, height - 0.42,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          facecolor=colors[color_key],
                          edgecolor=colors['package_border'],
                          linewidth=1.5)
    ax.add_patch(body)
    return y

def draw_component_with_params(ax, x, y, width, height, name, items, params, color_key, border_color):
    """绘制带参数的UML组件"""
    comp = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0,rounding_size=0.03",
                          facecolor=colors[color_key],
                          edgecolor=border_color,
                          linewidth=1.5)
    ax.add_patch(comp)
    
    # 标题
    ax.text(x + width/2, y + height - 0.22, name, ha='center', va='center',
            fontsize=9, fontweight='bold', color=border_color)
    
    # 分隔线
    ax.plot([x + 0.08, x + width - 0.08], [y + height - 0.38, y + height - 0.38], 
            color=border_color, linewidth=0.8)
    
    # 主项目
    item_y = y + height - 0.52
    for item in items:
        ax.text(x + 0.1, item_y, item, ha='left', va='center',
                fontsize=8, color=colors['text'])
        item_y -= 0.23
    
    # 参数 - 从底部往上排列
    param_y = y + 0.08
    for param in reversed(params):
        ax.text(x + 0.1, param_y, param, ha='left', va='center',
                fontsize=6.5, color='#555')
        param_y += 0.11
    
    return y

def draw_component_simple(ax, x, y, width, height, name, items, color_key, border_color):
    """绘制简单UML组件（无参数）"""
    comp = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0,rounding_size=0.03",
                          facecolor=colors[color_key],
                          edgecolor=border_color,
                          linewidth=1.5)
    ax.add_patch(comp)
    
    # 标题
    ax.text(x + width/2, y + height - 0.22, name, ha='center', va='center',
            fontsize=9, fontweight='bold', color=border_color)
    
    # 分隔线
    ax.plot([x + 0.08, x + width - 0.08], [y + height - 0.38, y + height - 0.38], 
            color=border_color, linewidth=0.8)
    
    # 项目 - 均匀分布
    item_spacing = (height - 0.5) / len(items)
    item_y = y + height - 0.52
    for item in items:
        ax.text(x + 0.1, item_y, item, ha='left', va='center',
                fontsize=8, color=colors['text'])
        item_y -= item_spacing
    
    return y

def draw_note(ax, x, y, text, width=1.45, height=0.46):
    """绘制注释"""
    note = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0,rounding_size=0.03",
                          facecolor=colors['note'],
                          edgecolor=colors['note_border'],
                          linewidth=1)
    ax.add_patch(note)
    # 折角
    fold = Polygon([(x + width - 0.12, y + height), 
                    (x + width, y + height - 0.12),
                    (x + width - 0.12, y + height - 0.12)],
                   facecolor='#ffe0b2', edgecolor=colors['note_border'], linewidth=1)
    ax.add_patch(fold)
    
    lines = text.split('\n')
    text_y = y + height - 0.1
    for line in lines:
        ax.text(x + 0.08, text_y, line, ha='left', va='top', 
                fontsize=6.5, color=colors['text'])
        text_y -= 0.11

# ==================== 开始绘制 ====================

# 标题
ax.text(4.135, 9.85, '图2-1 基于增强五维模型的月面巡视器自主行走数字孪生总体架构', 
        ha='center', va='center', fontsize=13, fontweight='bold', color=colors['title'])

# 核心公式
formula_box = FancyBboxPatch((1.3, 9.25), 5.67, 0.32,
                             boxstyle="round,pad=0,rounding_size=0.05",
                             facecolor='#fff8e1',
                             edgecolor='#ff8f00',
                             linewidth=2)
ax.add_patch(formula_box)
ax.text(4.135, 9.41, '增强五维模型 = (物理实体, 虚拟实体, 服务, 孪生数据, 连接)', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='#e65100')

# ========== 第1层：增强服务层 SsE ==========
# 增加层高度，给组件下移留出空间
service_layer_height = 1.75
service_layer_y = 7.3

draw_package(ax, 0.35, service_layer_y, 7.57, service_layer_height, 
             '增强服务层 (SsE)', 'service', colors['service_border'])

comp_width = 2.25
comp_height = 1.15
# 下移组件，与层标题保持充足间距
comp_y = service_layer_y + 0.1  # 从0.15增加到0.22

# 三个服务组件
draw_component_simple(ax, 0.5, comp_y, comp_width, comp_height,
               '自主感知服务',
               ['岩石检测 (SiaT-Hough)',
                '视觉定位 (ORB-SLAM3)',
                '威胁评估'],
               'service', colors['service_border'])

draw_component_simple(ax, 2.9, comp_y, comp_width, comp_height,
               '自主规划服务',
               ['全局规划器 (A*)',
                '局部规划器 (D3QN)',
                '路径优化器'],
               'service', colors['service_border'])

draw_component_simple(ax, 5.3, comp_y, comp_width, comp_height,
               '自主控制服务',
               ['运动控制器',
                '轮壤交互控制',
                '动力学仿真器'],
               'service', colors['service_border'])

draw_note(ax, 6.85, comp_y + 0.62, '成熟度: L1→L4\n自主智能决策', 0.9, 0.43)

# ========== 第2层：增强孪生数据层 DDE ==========
data_layer_height = 1.7
data_layer_y = 5.4

draw_package(ax, 0.35, data_layer_y, 7.57, data_layer_height, 
             '增强孪生数据层 (DDE)', 'data', colors['data_border'])

comp_y = data_layer_y + 0.078

draw_component_simple(ax, 0.5, comp_y, comp_width, comp_height,
               '数据同步服务',
               ['预测器 (KF/PF)',
                '时延补偿器',
                '一致性检验器'],
               'data', colors['data_border'])

draw_component_simple(ax, 2.9, comp_y, comp_width, comp_height,
               '数据管理服务',
               ['数据压缩器',
                '优先级调度器',
                '融合引擎'],
               'data', colors['data_border'])

draw_component_simple(ax, 5.3, comp_y, comp_width, comp_height,
               '边缘处理服务',
               ['特征提取器',
                '异常检测器',
                '本地存储器'],
               'data', colors['data_border'])

draw_note(ax, 6.85, comp_y + 0.62, '成熟度: L1→L3\n时延容忍同步', 0.9, 0.43)

# ========== 连接关系：服务层到数据层 ==========
for x in [1.62, 4.02, 6.42]:
    ax.plot([x, x], [7.3, 7.1], color='#666', linewidth=0.8, linestyle='--')
    ax.annotate('', xy=(x, 7.1), xytext=(x, 7.22),
                arrowprops=dict(arrowstyle='->', color='#666', lw=0.8))

# ========== 第3层：增强虚拟实体层 VEE ==========
vee_layer_height = 2.8
vee_layer_y = 2.35

draw_package(ax, 0.35, vee_layer_y, 7.57, vee_layer_height, 
             '增强虚拟实体层 (VEE)', 'component', colors['component_border'])

env_width = 3.35
env_height = 2.1
env_y = vee_layer_y + 0.175

env_items = [
    '地形模型 (DEM/TIN)',
    '月壤模型 (Bekker/DEM)',
    '光照模型 (BRDF)',
    '温度模型'
]

env_params = [
    '地形: 多分辨率DEM (1m/2m/10m)',
    '月壤: kc/kphi/phi/n 参数场',
    '光照: 太阳高度角/阴影分析',
    '温度: -180°C ~ +120°C'
]

draw_component_with_params(ax, 0.5, env_y, env_width, env_height,
               '环境数字孪生',
               env_items,
               env_params,
               'component', colors['component_border'])

dyn_items = [
    '运动学模型 (摇臂-转向架)',
    '轮壤交互模型',
    '多体动力学模型',
    '参数估计器'
]

dyn_params = [
    '运动学: D-H参数/正逆运动学',
    '轮壤: Bekker沉陷/剪切模型',
    '多体: 递归算法/低重力效应',
    '估计: RLS/EKF参数辨识'
]

draw_component_with_params(ax, 4.25, env_y, env_width, env_height,
               '动力学数字孪生',
               dyn_items,
               dyn_params,
               'component', colors['component_border'])

# 耦合关系
ax.annotate('', xy=(4.18, env_y + 1.2), xytext=(3.97, env_y + 1.2),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax.text(4.075, env_y + 1.28, '耦合', ha='center', va='bottom', fontsize=6.5, color='#666')

draw_note(ax, 6.8, env_y + 1.58, '成熟度: L1→L4\n多物理场耦合', 0.95, 0.43)

# ========== 连接关系：数据层到虚拟实体层 ==========
ax.plot([4.075, 4.075], [5.4, 5.2], color='#666', linewidth=0.8, linestyle='--')
ax.annotate('', xy=(4.075, 5.2), xytext=(4.075, 5.32),
            arrowprops=dict(arrowstyle='->', color='#666', lw=0.8))
ax.text(4.18, 5.25, '驱动', ha='left', va='center', fontsize=6, color='#666')


# 保存
plt.tight_layout()
plt.savefig('output/figure_2_1_uml_v13.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.1)
plt.savefig('output/figure_2_1_uml_v13.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)
print("UML架构图v13已保存 - 美感优化完成")

plt.close()

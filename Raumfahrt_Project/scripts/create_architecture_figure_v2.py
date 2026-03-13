#!/usr/bin/env python3
"""
创建论文图2-1：基于增强五维模型的月面巡视器自主行走数字孪生总体架构 V2
优化版本 - 更大的字体和更清晰的布局
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建超大画布 - A0尺寸
fig = plt.figure(figsize=(48, 64), facecolor='#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# 定义颜色方案 - 五维度配色
colors = {
    'PEE': '#bbdefb',      # 浅蓝 - 物理实体
    'PEE_border': '#1565c0',
    'VEE': '#e1bee7',      # 浅紫 - 虚拟实体
    'VEE_border': '#6a1b9a',
    'DDE': '#ffe0b2',      # 浅橙 - 孪生数据
    'DDE_border': '#e65100',
    'SsE': '#c8e6c9',      # 浅绿 - 服务
    'SsE_border': '#2e7d32',
    'CNE': '#f8bbd9',      # 浅粉 - 连接
    'CNE_border': '#c2185b',
    'title': '#212121',
    'subtitle': '#424242',
    'text': '#333333',
    'highlight': '#d84315',
    'white': '#ffffff'
}

def draw_box(ax, x, y, width, height, text, color_key, fontsize=14, text_color='#333', 
             border_width=3, radius=0.015, alpha=1.0, bold=False):
    """绘制带文本的圆角矩形框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=colors[color_key],
                         edgecolor=colors[f'{color_key}_border'],
                         linewidth=border_width,
                         alpha=alpha)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, wrap=True,
            fontweight=weight, linespacing=1.3)
    return box

def draw_level_badge(ax, x, y, level_from, level_to, color):
    """绘制成熟度等级徽章"""
    badge = FancyBboxPatch((x, y), 0.08, 0.025,
                           boxstyle="round,pad=0,rounding_size=0.008",
                           facecolor=color,
                           edgecolor='white',
                           linewidth=2)
    ax.add_patch(badge)
    ax.text(x + 0.04, y + 0.0125, f'L{level_from} to L{level_to}', 
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# 创建主坐标轴
ax = fig.add_axes([0.015, 0.01, 0.97, 0.98])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_facecolor('#f8f9fa')

# ============================================
# 主标题
# ============================================
ax.text(0.5, 0.990, '图2-1 基于增强五维模型的月面巡视器自主行走数字孪生总体架构', 
        ha='center', va='top', fontsize=42, fontweight='bold', color=colors['title'])
ax.text(0.5, 0.982, 'Enhanced Five-Dimension Digital Twin Architecture for Lunar Rover Autonomous Navigation', 
        ha='center', va='top', fontsize=22, color=colors['subtitle'], style='italic')

# 核心公式
formula_box = FancyBboxPatch((0.30, 0.965), 0.40, 0.012,
                              boxstyle="round,pad=0,rounding_size=0.005",
                              facecolor=colors['DDE'],
                              edgecolor=colors['DDE_border'],
                              linewidth=3)
ax.add_patch(formula_box)
ax.text(0.50, 0.971, 'EFDT = (PEE, VEE, SsE, DDE, CNE)', ha='center', va='center',
        fontsize=20, color=colors['DDE_border'], fontweight='bold')

# ============================================
# 第一层：增强服务层 SsE
# ============================================
ss_e_y = 0.89
draw_box(ax, 0.03, ss_e_y, 0.94, 0.065, '', 'SsE', border_width=4, radius=0.015)
draw_box(ax, 0.30, 0.948, 0.40, 0.014, 
         '增强服务层 SsE (Service) - 自主智能服务', 'SsE', fontsize=22, 
         text_color='white', border_width=0, bold=True)
draw_level_badge(ax, 0.71, 0.947, 1, 4, colors['SsE_border'])

# 自主感知服务
ax.text(0.15, 0.935, '自主感知服务 (第5章)', ha='center', va='center',
        fontsize=18, fontweight='bold', color=colors['SsE_border'])
perception_text = '''5.1 月面岩石识别需求与挑战
  - 工程需求分析
  - 月面场景关键挑战(光照/地形/计算资源)

5.2 数字孪生平台下的岩石识别框架
  - 虚实协同感知框架
  - 数据生成与验证环境

5.3 增强型ORB-SLAM2系统
  - SiaT-Hough轻量级岩石感知算法
  - Cesium-ORB-SLAM3协同语义重建

5.4 基于岩石识别的避障策略
  - 多级威胁评估模型
  - 动态避障决策机制'''
ax.text(0.03, 0.925, perception_text, ha='left', va='top',
        fontsize=12, color=colors['text'], linespacing=1.4, )

# 自主规划服务
ax.text(0.50, 0.935, '自主规划服务 (第6章)', ha='center', va='center',
        fontsize=18, fontweight='bold', color=colors['SsE_border'])
planning_text = '''6.1 路径规划问题描述
  - 状态/动作/环境定义
  - 约束条件(避障/运动学/动力学)
  - 优化目标与性能指标

6.2 数字孪生环境下的路径规划框架
  - 分层规划架构
  - 全局-局部协同机制

6.3 A*-D3QN-Opt混合路径规划算法
  - 改进A*全局规划(8邻域/能耗代价/平滑)
  - D3QN深度强化学习(Dueling+Double DQN)
  - 混合算法融合策略

6.4 算法验证与性能分析'''
ax.text(0.35, 0.925, planning_text, ha='left', va='top',
        fontsize=12, color=colors['text'], linespacing=1.4, )

# 自主控制服务
ax.text(0.82, 0.935, '自主控制服务 (第4章)', ha='center', va='center',
        fontsize=18, fontweight='bold', color=colors['SsE_border'])
control_text = '''4.2 多体动力学模型构建
  - 运动学建模(正/逆运动学)
  - 轮-壤交互力学建模(Bekker理论)
  - 多体动力学建模(ADAMS模型)

4.3 动力学模型验证与校准
  - RLS递归最小二乘参数辨识
  - EKF扩展卡尔曼滤波估计

4.4 动力学数字孪生实时仿真'''
ax.text(0.68, 0.925, control_text, ha='left', va='top',
        fontsize=12, color=colors['text'], linespacing=1.4, )

# ============================================
# 第二层：增强孪生数据层 DDE
# ============================================
dde_y = 0.805
draw_box(ax, 0.03, dde_y, 0.94, 0.075, '', 'DDE', border_width=4, radius=0.015)
draw_box(ax, 0.30, 0.875, 0.40, 0.014, 
         '增强孪生数据层 DDE (Twin Data) - 时延容忍同步', 'DDE', fontsize=22, 
         text_color='white', border_width=0, bold=True)
draw_level_badge(ax, 0.71, 0.874, 1, 3, colors['DDE_border'])

dde_text = '''时延容忍数据同步          数据完整性保障            自主数据处理
  - 预测-校正方法           - 数据压缩机制             - 边缘计算能力
  - 卡尔曼滤波/粒子滤波     - 优先级调度机制           - 特征提取与异常检测
  - 状态最优估计            - 数据融合与插值重建       - 本地自主决策支持'''
ax.text(0.05, 0.865, dde_text, ha='left', va='top',
        fontsize=14, color=colors['text'], linespacing=1.5, )

# ============================================
# 第三层：增强虚拟实体层 VEE
# ============================================
vee_y = 0.685
draw_box(ax, 0.03, vee_y, 0.94, 0.110, '', 'VEE', border_width=4, radius=0.015)
draw_box(ax, 0.30, 0.790, 0.40, 0.014, 
         '增强虚拟实体层 VEE (Virtual Entity) - 多物理场耦合仿真', 'VEE', fontsize=22, 
         text_color='white', border_width=0, bold=True)
draw_level_badge(ax, 0.71, 0.789, 1, 4, colors['VEE_border'])

# 第3章 月面环境数字孪生
ax.text(0.25, 0.775, '第3章 月面环境数字孪生建模与仿真', ha='center', va='center',
        fontsize=20, fontweight='bold', color=colors['VEE_border'])
env_text = '''3.2 月面地形数字孪生建模
  - 基于嫦娥6号DEM数据的高程图构建
  - 多分辨率地形表示(全局-局部融合)
  - 地形特征提取(坡度/粗糙度/可通行性)

3.3 月壤环境特性数字孪生建模
  - Bekker-Wong土壤力学模型
  - 离散元方法(DEM)高保真月壤仿真
  - 月壤物理参数数据库(粘聚模量/摩擦模量/内摩擦角)

3.4 月面光照与温度环境仿真
  - 动态光照建模(太阳高度角变化/地球反照/地形遮挡)
  - 温度场建模(太阳辐射/月面热惯性/阴影效应)'''
ax.text(0.05, 0.765, env_text, ha='left', va='top',
        fontsize=12, color=colors['text'], linespacing=1.4, )

# 第4章 动力学数字孪生
ax.text(0.72, 0.775, '第4章 月面巡视器动力学数字孪生建模', ha='center', va='center',
        fontsize=20, fontweight='bold', color=colors['VEE_border'])
dyn_text = '''4.2 多体动力学模型构建
  - 运动学建模: 正/逆运动学, 六轮摇臂-转向架结构
  - 轮-壤交互力学建模: Bekker沉陷模型, Wong剪切模型
  - 多体动力学建模: ADAMS模型, 多物理场耦合

4.3 动力学模型验证与校准
  - RLS递归最小二乘参数辨识
  - EKF扩展卡尔曼滤波状态估计
  - 虚实同步一致性验证'''
ax.text(0.52, 0.765, dyn_text, ha='left', va='top',
        fontsize=12, color=colors['text'], linespacing=1.4, )

# ============================================
# 第四层：增强物理实体层 PEE
# ============================================
pee_y = 0.555
draw_box(ax, 0.03, pee_y, 0.94, 0.120, '', 'PEE', border_width=4, radius=0.015)
draw_box(ax, 0.30, 0.670, 0.40, 0.014, 
         '增强物理实体层 PEE (Physical Entity) - 极端环境适应性建模', 'PEE', fontsize=22, 
         text_color='white', border_width=0, bold=True)
draw_level_badge(ax, 0.71, 0.669, 1, 2, colors['PEE_border'])

# PEE 四列布局
pee_items = [
    ('极端温度环境适应性建模', 
     '- 温度-性能耦合建模\n- 电池/电机/电子设备\n  温度影响系数模型\n- 工作温度: -180C至+120C',
     0.05),
    ('真空环境效应建模', 
     '- 真空-材料相互作用模型\n- 材料出气/润滑失效\n- 固体润滑模型\n- 辐射散热模型',
     0.28),
    ('辐射环境可靠性建模', 
     '- 辐射剂量-故障率关系\n- 单粒子效应(SEE)预测\n- 辐射屏蔽模型\n- 故障容错机制',
     0.51),
    ('月面巡视器实体配置', 
     '- 六轮摇臂-转向架结构\n- 质量: 140kg, 轮径: 0.1525m\n- 传感器: 立体视觉/激光雷达\n- IMU/太阳敏感器/里程计',
     0.74)
]

for title, content, x_pos in pee_items:
    ax.text(x_pos + 0.10, 0.655, title, ha='center', va='center',
            fontsize=14, fontweight='bold', color=colors['PEE_border'])
    ax.text(x_pos, 0.640, content, ha='left', va='top',
            fontsize=11, color=colors['text'], linespacing=1.4, )

# ============================================
# 第五层：增强连接层 CNE
# ============================================
cne_y = 0.465
draw_box(ax, 0.03, cne_y, 0.94, 0.080, '', 'CNE', border_width=4, radius=0.015)
draw_box(ax, 0.30, 0.540, 0.40, 0.014, 
         '增强连接层 CNE (Connection) - 异步通信与自主协同', 'CNE', fontsize=22, 
         text_color='white', border_width=0, bold=True)
draw_level_badge(ax, 0.71, 0.539, 1, 3, colors['CNE_border'])

cne_text = '''异步通信机制                    自主协同协议                    可靠传输保障
  - 消息队列架构                 - 通信正常/延迟/中断三种状态规则      - 自适应传输协议
  - 发布-订阅模式                - 地面预演-月面执行异步自主模式        - 前向纠错(FEC)
  - 延迟容忍数据传输             - 日凌期间通信中断自主运行             - 自动重传请求(ARQ)'''
ax.text(0.05, 0.525, cne_text, ha='left', va='top',
        fontsize=13, color=colors['text'], linespacing=1.5, )

# ============================================
# 多维协同机制
# ============================================
ax.text(0.5, 0.450, '增强五维模型的多维协同机制', ha='center', va='center',
        fontsize=26, fontweight='bold', color=colors['title'])

coop_y = 0.365
coop_box = FancyBboxPatch((0.03, coop_y), 0.94, 0.075,
                          boxstyle="round,pad=0,rounding_size=0.01",
                          facecolor=colors['DDE'],
                          edgecolor=colors['DDE_border'],
                          linewidth=2,
                          alpha=0.3)
ax.add_patch(coop_box)

coop_text = '''数据-模型协同                              模型-服务协同                              服务-物理协同                              数据中枢协同
PEE -> DDE -> VEE                         VEE <-> SsE                               SsE -> PEE                                DDE核心数据中枢
"物理-数据-模型"递进协同                   "仿真-决策-反馈"闭环协同                   "决策-执行-监测"虚实协同                   数据驱动的全局协同'''
ax.text(0.05, 0.425, coop_text, ha='left', va='top',
        fontsize=13, color=colors['text'], linespacing=1.6, fontweight='bold')

# ============================================
# 成熟度评价框架
# ============================================
ax.text(0.5, 0.345, '数字孪生成熟度评价框架 (GB/T 46237-2025)', ha='center', va='center',
        fontsize=26, fontweight='bold', color=colors['title'])

maturity_y = 0.245
mat_box = FancyBboxPatch((0.03, maturity_y), 0.94, 0.090,
                         boxstyle="round,pad=0,rounding_size=0.01",
                         facecolor=colors['VEE'],
                         edgecolor=colors['VEE_border'],
                         linewidth=2,
                         alpha=0.2)
ax.add_patch(mat_box)

# L0-L5等级
levels = [
    ('L0 以虚仿实', '虚拟模型仅能仿真\n基本行为,未经验证', 0.05, '#9e9e9e'),
    ('L1 以虚映实', '实时反映物理实体状态\n实现"虚实映射"', 0.20, '#757575'),
    ('L2 以虚控实', '控制物理实体行为\n实现"虚实交互"', 0.35, '#616161'),
    ('L3 以虚预实', '预测物理实体未来状态\n实现"虚实预测"', 0.50, '#424242'),
    ('L4 以虚优实', '优化物理实体性能\n实现"虚实优化"', 0.65, '#212121'),
    ('L5 虚实共生', '虚拟与物理完全融合\n协同进化', 0.80, '#000000')
]

for title, desc, x_pos, color in levels:
    level_box = FancyBboxPatch((x_pos, maturity_y + 0.01), 0.14, 0.068,
                               boxstyle="round,pad=0,rounding_size=0.005",
                               facecolor=color,
                               edgecolor='white',
                               linewidth=2,
                               alpha=0.9)
    ax.add_patch(level_box)
    ax.text(x_pos + 0.07, maturity_y + 0.055, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(x_pos + 0.07, maturity_y + 0.028, desc, ha='center', va='center',
            fontsize=9, color='white', linespacing=1.2)

# ============================================
# 分章评价实施内容
# ============================================
ax.text(0.5, 0.225, '分章成熟度评价实施内容', ha='center', va='center',
        fontsize=26, fontweight='bold', color=colors['title'])

chapter_y = 0.115
chap_box = FancyBboxPatch((0.03, chapter_y), 0.94, 0.100,
                          boxstyle="round,pad=0,rounding_size=0.01",
                          facecolor=colors['PEE'],
                          edgecolor=colors['PEE_border'],
                          linewidth=2,
                          alpha=0.2)
ax.add_patch(chap_box)

chapters = [
    ('第3章 月面环境建模', 'VEE: L1->L3\nDDE: L1->L2', 
     '地形建模保真度\n月壤建模准确度\n光照建模精度',
     0.06),
    ('第4章 动力学建模', 'PEE: L1->L2\nVEE: L1->L4\nDDE: L1->L3', 
     '极端环境适应性\n运动学建模精度\n轮壤交互准确性',
     0.29),
    ('第5章 岩石感知与避障', 'SsE: L1->L3', 
     '岩石检测准确率\nSLAM定位精度\n威胁评估有效性',
     0.52),
    ('第6章 路径规划', 'SsE: L1->L4\nCNE: L1->L3', 
     '全局规划优化效果\n局部规划响应速度\n异步协同效率',
     0.75)
]

for title, level, content, x_pos in chapters:
    ax.text(x_pos + 0.10, chapter_y + 0.085, title, ha='center', va='center',
            fontsize=14, fontweight='bold', color=colors['PEE_border'])
    ax.text(x_pos + 0.10, chapter_y + 0.065, level, ha='center', va='center',
            fontsize=12, color=colors['highlight'], fontweight='bold', linespacing=1.3)
    ax.text(x_pos + 0.10, chapter_y + 0.035, content, ha='center', va='center',
            fontsize=10, color=colors['text'], linespacing=1.3)
    ax.text(x_pos + 0.10, chapter_y + 0.015, '[详细评价指标见各章]', ha='center', va='center',
            fontsize=9, color='#666', style='italic')

# ============================================
# 底部说明
# ============================================
ax.text(0.5, 0.095, '核心创新: 面向月面巡视器自主行走的增强五维数字孪生模型 (EFDT)', 
        ha='center', va='center', fontsize=22, fontweight='bold', 
        color=colors['highlight'])

ax.text(0.5, 0.075, '研究逻辑: 环境认知(第3章) -> 运动预测(第4章) -> 智能决策(第5-6章)', 
        ha='center', va='center', fontsize=18, color=colors['subtitle'])

ax.text(0.5, 0.055, '系统性解决三大挑战: 月面极端环境适应性 | 地-月通信受限性 | 任务执行自主性', 
        ha='center', va='center', fontsize=16, color=colors['text'])

ax.text(0.5, 0.035, '论文结构: 第2章理论框架 -> 第3章环境建模 -> 第4章动力学建模 -> 第5章感知避障 -> 第6章路径规划', 
        ha='center', va='center', fontsize=14, color='#666', style='italic')

# 图注
ax.text(0.5, 0.015, '图2-1 基于增强五维模型的月面巡视器自主行走数字孪生总体架构 | 第2章 理论框架', 
        ha='center', va='center', fontsize=14, color='#888', style='italic')

# ============================================
# 保存图片
# ============================================
plt.savefig('output/figure_2_1_architecture_v2.png', dpi=200, bbox_inches='tight', 
            facecolor='#f8f9fa', edgecolor='none', pad_inches=0.2)
plt.savefig('output/figure_2_1_architecture_v2.pdf', bbox_inches='tight', 
            facecolor='#f8f9fa', edgecolor='none', pad_inches=0.2)
print("架构图已保存到 output/figure_2_1_architecture_v2.png 和 .pdf")

plt.close()
print("完成!")

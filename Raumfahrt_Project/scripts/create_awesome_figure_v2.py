#!/usr/bin/env python3
"""
创建论文用的超复杂精美图表 V2 - 修复字体问题
包含：3D地形、多物理场、路径规划、动力学分析
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Arrow, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建随机种子确保可复现
np.random.seed(42)

# ============================================
# 生成复杂地形数据
# ============================================
resolution = 0.5  # m/pixel
map_size = (200, 200)  # 100m x 100m

# 创建高程图 - 使用多种噪声叠加
x = np.linspace(0, 100, map_size[0])
y = np.linspace(0, 100, map_size[1])
X, Y = np.meshgrid(x, y)

# 复杂地形生成
elevation = (
    2 * np.sin(X/15) * np.cos(Y/12) +  # 大尺度起伏
    1.5 * np.sin(X/8 + Y/10) +
    0.8 * np.random.randn(*map_size) * 0.3 +  # 粗糙度
    3 * np.exp(-((X-30)**2 + (Y-70)**2)/200) +  # 小山丘
    2 * np.exp(-((X-70)**2 + (Y-30)**2)/150)    # 另一个山丘
)
elevation = np.maximum(elevation, 0)  # 确保非负

# 坡度计算
slope_x, slope_y = np.gradient(elevation, resolution)
slope = np.sqrt(slope_x**2 + slope_y**2)
slope_deg = np.degrees(np.arctan(slope))

# 粗糙度（局部高程变化）
local_mean = np.zeros_like(elevation)
for i in range(2, elevation.shape[0]-2):
    for j in range(2, elevation.shape[1]-2):
        local_mean[i,j] = np.mean(elevation[i-2:i+3, j-2:j+3])
roughness = np.abs(elevation - local_mean)

# 可通行性（综合坡度、粗糙度）
traversability = np.exp(-slope_deg/15) * np.exp(-roughness/0.5)

# 语义分割 / 土壤类型
soil_types = np.zeros(map_size, dtype=int)
soil_types[elevation < 1.5] = 0  # 松软月壤
soil_types[(elevation >= 1.5) & (elevation < 4)] = 1  # 压实月壤
soil_types[elevation >= 4] = 2  # 岩石
random_mask = np.random.rand(*map_size) < 0.05
soil_types[random_mask] = np.random.choice([0, 1, 2], size=np.sum(random_mask))

# 物理参数映射
kc_map = np.where(soil_types == 0, 1.4e3, np.where(soil_types == 1, 2.9e4, 1e8))
kphi_map = np.where(soil_types == 0, 8.2e5, np.where(soil_types == 1, 1.5e6, 1e8))
phi_map = np.where(soil_types == 0, 30, np.where(soil_types == 1, 35, 45))

# 创建障碍物
obstacles = []
np.random.seed(123)
for _ in range(25):
    ox = np.random.uniform(10, 90)
    oy = np.random.uniform(10, 90)
    orad = np.random.uniform(1.5, 4)
    obstacles.append((ox, oy, orad))

# 标记障碍物区域
obstacle_map = np.zeros(map_size)
for ox, oy, orad in obstacles:
    dist = np.sqrt((X - ox)**2 + (Y - oy)**2)
    obstacle_map[dist < orad] = 1

# ============================================
# 生成路径数据
# ============================================
start = np.array([10, 10])
goal = np.array([90, 85])

n_points = 150
t = np.linspace(0, 1, n_points)
path_x = start[0] + (goal[0] - start[0]) * t + 15 * np.sin(t * np.pi * 2) * np.sin(t * np.pi * 3)
path_y = start[1] + (goal[1] - start[1]) * t + 10 * np.cos(t * np.pi * 1.5) * np.sin(t * np.pi * 2)

path_x = np.clip(path_x, 5, 95)
path_y = np.clip(path_y, 5, 95)

# 动力学仿真数据
path_length = np.zeros(n_points)
for i in range(1, n_points):
    path_length[i] = path_length[i-1] + np.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2)

velocity = 0.3 + 0.2 * traversability[
    np.clip((path_y/resolution).astype(int), 0, map_size[0]-1),
    np.clip((path_x/resolution).astype(int), 0, map_size[1]-1)
]
velocity = np.convolve(velocity, np.ones(5)/5, mode='same')

slip_ratio = 0.05 + 0.1 * (1 - traversability[
    np.clip((path_y/resolution).astype(int), 0, map_size[0]-1),
    np.clip((path_x/resolution).astype(int), 0, map_size[1]-1)
])

sinkage = 0.02 + 0.08 * slip_ratio

power = 50 + 100 * slip_ratio + 30 * slope[
    np.clip((path_y/resolution).astype(int), 0, map_size[0]-1),
    np.clip((path_x/resolution).astype(int), 0, map_size[1]-1)
]
energy = np.cumsum(power) * 0.1

# ============================================
# 创建超复杂图表
# ============================================
fig = plt.figure(figsize=(26, 22), facecolor='#0d1117')
fig.patch.set_facecolor('#0d1117')

# 创建复杂的gridspec布局
gs = gridspec.GridSpec(4, 4, height_ratios=[1.2, 1, 1, 0.8], 
                       width_ratios=[1, 1, 1, 0.6],
                       hspace=0.22, wspace=0.22,
                       left=0.035, right=0.965, top=0.94, bottom=0.035)

# 自定义颜色映射
terrain_cmap = LinearSegmentedColormap.from_list('terrain', 
    ['#1a237e', '#283593', '#3949ab', '#5e35b1', '#8e24aa', 
     '#d81b60', '#f4511e', '#fb8c00', '#ffb300', '#fdd835', '#fff176'])

# ============================================
# 子图1: 3D地形 + 路径 (左上大图)
# ============================================
ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d', facecolor='#0d1117')
ax1.set_facecolor('#0d1117')

# 绘制3D地形
surf = ax1.plot_surface(X, Y, elevation, cmap=terrain_cmap, 
                        alpha=0.9, linewidth=0, antialiased=True,
                        rstride=4, cstride=4, shade=True)

# 在3D地形上绘制路径
path_z = elevation[
    np.clip((path_y/resolution).astype(int), 0, map_size[0]-1),
    np.clip((path_x/resolution).astype(int), 0, map_size[1]-1)
] + 0.5

# 绘制3D路径（带颜色渐变）
for i in range(len(path_x)-1):
    color = plt.cm.hot(velocity[i] / velocity.max())
    ax1.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
             [path_z[i], path_z[i+1]], color=color, linewidth=3, alpha=0.95)

# 起点和终点
ax1.scatter(*start, path_z[0]+1.5, c='#00ff00', s=300, marker='*', 
            edgecolors='white', linewidths=2, label='Start', zorder=10)
ax1.scatter(*goal, path_z[-1]+1.5, c='#ff1744', s=300, marker='X', 
            edgecolors='white', linewidths=2, label='Goal', zorder=10)

# 绘制障碍物为3D圆柱
for ox, oy, orad in obstacles[:10]:
    oz = elevation[int(oy/resolution), int(ox/resolution)]
    theta = np.linspace(0, 2*np.pi, 15)
    z_cyl = np.linspace(oz, oz+3, 8)
    theta, z_cyl = np.meshgrid(theta, z_cyl)
    x_cyl = ox + orad * np.cos(theta)
    y_cyl = oy + orad * np.sin(theta)
    ax1.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.5, color='#ff5252', shade=False)

ax1.set_xlabel('X (m)', color='white', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y (m)', color='white', fontsize=11, fontweight='bold')
ax1.set_zlabel('Elevation (m)', color='white', fontsize=11, fontweight='bold')
ax1.set_title('3D Lunar Terrain with Optimal Path\n(A* + D3QN Hybrid Planning)', 
              color='white', fontsize=15, fontweight='bold', pad=20)
ax1.tick_params(colors='white', labelsize=9)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('#444')
ax1.yaxis.pane.set_edgecolor('#444')
ax1.zaxis.pane.set_edgecolor('#444')
ax1.xaxis.pane.set_alpha(0.1)
ax1.yaxis.pane.set_alpha(0.1)
ax1.zaxis.pane.set_alpha(0.1)
ax1.view_init(elev=30, azim=55)

# ============================================
# 子图2: 高程热力图 (右上)
# ============================================
ax2 = fig.add_subplot(gs[0, 2], facecolor='#0d1117')
im2 = ax2.imshow(elevation, cmap='terrain', origin='lower', extent=[0, 100, 0, 100])
ax2.plot(path_x, path_y, '#ff1744', linewidth=2.5, alpha=0.9)
ax2.scatter(*start, c='#00ff00', s=200, marker='*', edgecolors='white', linewidths=2, zorder=10)
ax2.scatter(*goal, c='#ff1744', s=200, marker='X', edgecolors='white', linewidths=2, zorder=10)
for ox, oy, orad in obstacles:
    circle = Circle((ox, oy), orad, fill=False, color='#ff5252', linewidth=2, alpha=0.8)
    ax2.add_patch(circle)
ax2.set_title('Elevation Map (m)', color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('X (m)', color='white', fontsize=10)
ax2.set_ylabel('Y (m)', color='white', fontsize=10)
ax2.tick_params(colors='white')
ax2.set_facecolor('#0d1117')
for spine in ax2.spines.values():
    spine.set_color('#444')
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors='white')
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

# ============================================
# 子图3: 坡度图 (右中上)
# ============================================
ax3 = fig.add_subplot(gs[0, 3], facecolor='#0d1117')
im3 = ax3.imshow(slope_deg, cmap='YlOrRd', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=30)
ax3.plot(path_x, path_y, '#00e5ff', linewidth=2.5, alpha=0.9)
ax3.set_title('Slope (degrees)', color='white', fontsize=12, fontweight='bold')
ax3.set_xlabel('X (m)', color='white', fontsize=10)
ax3.tick_params(colors='white')
ax3.set_facecolor('#0d1117')
for spine in ax3.spines.values():
    spine.set_color('#444')
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.ax.tick_params(colors='white')
plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')

# ============================================
# 子图4: 可通行性 (第二行左)
# ============================================
ax4 = fig.add_subplot(gs[1, 2], facecolor='#0d1117')
im4 = ax4.imshow(traversability, cmap='RdYlGn', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
ax4.plot(path_x, path_y, 'white', linewidth=3, alpha=0.95)
ax4.scatter(*start, c='#00ff00', s=200, marker='*', edgecolors='white', linewidths=2, zorder=10)
ax4.scatter(*goal, c='#ff1744', s=200, marker='X', edgecolors='white', linewidths=2, zorder=10)
ax4.set_title('Traversability Index', color='white', fontsize=12, fontweight='bold')
ax4.set_xlabel('X (m)', color='white', fontsize=10)
ax4.tick_params(colors='white')
ax4.set_facecolor('#0d1117')
for spine in ax4.spines.values():
    spine.set_color('#444')
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.ax.tick_params(colors='white')
plt.setp(plt.getp(cbar4.ax.axes, 'yticklabels'), color='white')

# ============================================
# 子图5: 土壤类型/物理参数 (第二行右)
# ============================================
ax5 = fig.add_subplot(gs[1, 3], facecolor='#0d1117')
soil_colors = np.array([[0.6, 0.4, 0.2], [0.85, 0.65, 0.35], [0.4, 0.4, 0.45]])
soil_rgb = soil_colors[soil_types]
im5 = ax5.imshow(soil_rgb, origin='lower', extent=[0, 100, 0, 100])
ax5.plot(path_x, path_y, '#00e5ff', linewidth=2.5, alpha=0.9)
ax5.set_title('Soil Types\n(0:Regolith 1:Firm 2:Rock)', color='white', fontsize=11, fontweight='bold')
ax5.set_xlabel('X (m)', color='white', fontsize=10)
ax5.tick_params(colors='white')
ax5.set_facecolor('#0d1117')
for spine in ax5.spines.values():
    spine.set_color('#444')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=soil_colors[0], edgecolor='white', label='Regolith'),
                   Patch(facecolor=soil_colors[1], edgecolor='white', label='Firm'),
                   Patch(facecolor=soil_colors[2], edgecolor='white', label='Rock')]
ax5.legend(handles=legend_elements, loc='lower right', facecolor='#1a1a2e', 
           edgecolor='white', labelcolor='white', fontsize=9)

# ============================================
# 子图6: kc参数 (第三行左)
# ============================================
ax6 = fig.add_subplot(gs[2, 0], facecolor='#0d1117')
# 归一化显示kc
kc_normalized = (kc_map - kc_map.min()) / (kc_map.max() - kc_map.min())
im6 = ax6.imshow(kc_normalized, cmap='plasma', origin='lower', extent=[0, 100, 0, 100])
ax6.plot(path_x, path_y, 'white', linewidth=2.5, alpha=0.9)
ax6.set_title('Cohesive Modulus kc (Normalized)\nBekker Soil Mechanics', 
              color='white', fontsize=11, fontweight='bold')
ax6.set_xlabel('X (m)', color='white', fontsize=10)
ax6.set_ylabel('Y (m)', color='white', fontsize=10)
ax6.tick_params(colors='white')
ax6.set_facecolor('#0d1117')
for spine in ax6.spines.values():
    spine.set_color('#444')
cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cbar6.ax.tick_params(colors='white')
plt.setp(plt.getp(cbar6.ax.axes, 'yticklabels'), color='white')
cbar6.set_label('Normalized kc', color='white')

# ============================================
# 子图7: kphi参数 (第三行中)
# ============================================
ax7 = fig.add_subplot(gs[2, 1], facecolor='#0d1117')
kphi_normalized = (kphi_map - kphi_map.min()) / (kphi_map.max() - kphi_map.min())
im7 = ax7.imshow(kphi_normalized, cmap='inferno', origin='lower', extent=[0, 100, 0, 100])
ax7.plot(path_x, path_y, '#00e5ff', linewidth=2.5, alpha=0.9)
ax7.set_title('Frictional Modulus kphi (Normalized)\nBekker Soil Mechanics', 
              color='white', fontsize=11, fontweight='bold')
ax7.set_xlabel('X (m)', color='white', fontsize=10)
ax7.tick_params(colors='white')
ax7.set_facecolor('#0d1117')
for spine in ax7.spines.values():
    spine.set_color('#444')
cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
cbar7.ax.tick_params(colors='white')
plt.setp(plt.getp(cbar7.ax.axes, 'yticklabels'), color='white')
cbar7.set_label('Normalized kphi', color='white')

# ============================================
# 子图8: 内摩擦角 (第三行右)
# ============================================
ax8 = fig.add_subplot(gs[2, 2], facecolor='#0d1117')
im8 = ax8.imshow(phi_map, cmap='magma', origin='lower', extent=[0, 100, 0, 100], vmin=25, vmax=50)
ax8.plot(path_x, path_y, '#00e5ff', linewidth=2.5, alpha=0.9)
ax8.set_title('Internal Friction Angle phi (deg)\nShear Strength Parameter', 
              color='white', fontsize=11, fontweight='bold')
ax8.set_xlabel('X (m)', color='white', fontsize=10)
ax8.tick_params(colors='white')
ax8.set_facecolor('#0d1117')
for spine in ax8.spines.values():
    spine.set_color('#444')
cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
cbar8.ax.tick_params(colors='white')
plt.setp(plt.getp(cbar8.ax.axes, 'yticklabels'), color='white')

# ============================================
# 子图9: 动力学参数时序 (第三行最右)
# ============================================
ax9 = fig.add_subplot(gs[2, 3], facecolor='#0d1117')
line1, = ax9.plot(path_length, velocity, '#00e676', linewidth=2.5, label='Velocity (m/s)')
ax9_twin = ax9.twinx()
line2, = ax9_twin.plot(path_length, slip_ratio*100, '#ff9100', linewidth=2.5, linestyle='--', label='Slip Ratio (%)')
ax9.set_xlabel('Path Length (m)', color='white', fontsize=10)
ax9.set_ylabel('Velocity (m/s)', color='#00e676', fontsize=10, fontweight='bold')
ax9_twin.set_ylabel('Slip Ratio (%)', color='#ff9100', fontsize=10, fontweight='bold')
ax9.tick_params(colors='white')
ax9_twin.tick_params(colors='#ff9100')
ax9.set_facecolor('#0d1117')
ax9_twin.set_facecolor('#0d1117')
for spine in ax9.spines.values():
    spine.set_color('#444')
ax9_twin.spines['right'].set_color('#ff9100')
ax9.set_title('Dynamics Along Path\nWheel-Soil Interaction', color='white', fontsize=11, fontweight='bold')
ax9.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
ax9_twin.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
ax9.grid(True, alpha=0.2, color='gray')

# ============================================
# 子图10: 沉陷量 (第四行左)
# ============================================
ax10 = fig.add_subplot(gs[3, 0], facecolor='#0d1117')
ax10.fill_between(path_length, sinkage*1000, alpha=0.6, color='#00bcd4')
ax10.plot(path_length, sinkage*1000, '#00bcd4', linewidth=2.5)
ax10.set_xlabel('Path Length (m)', color='white', fontsize=10)
ax10.set_ylabel('Sinkage (mm)', color='#00bcd4', fontsize=10, fontweight='bold')
ax10.set_title('Wheel Sinkage', color='white', fontsize=11, fontweight='bold')
ax10.tick_params(colors='white')
ax10.set_facecolor('#0d1117')
for spine in ax10.spines.values():
    spine.set_color('#444')
ax10.spines['left'].set_color('#00bcd4')
ax10.grid(True, alpha=0.2, color='gray')

# ============================================
# 子图11: 能耗 (第四行中)
# ============================================
ax11 = fig.add_subplot(gs[3, 1], facecolor='#0d1117')
ax11.fill_between(path_length, energy/1000, alpha=0.6, color='#ffeb3b')
ax11.plot(path_length, energy/1000, '#ffeb3b', linewidth=2.5)
ax11.set_xlabel('Path Length (m)', color='white', fontsize=10)
ax11.set_ylabel('Energy (kJ)', color='#ffeb3b', fontsize=10, fontweight='bold')
ax11.set_title('Cumulative Energy Consumption', color='white', fontsize=11, fontweight='bold')
ax11.tick_params(colors='white')
ax11.set_facecolor('#0d1117')
for spine in ax11.spines.values():
    spine.set_color('#444')
ax11.spines['left'].set_color('#ffeb3b')
ax11.grid(True, alpha=0.2, color='gray')

# ============================================
# 子图12: 功率分布 (第四行右)
# ============================================
ax12 = fig.add_subplot(gs[3, 2], facecolor='#0d1117')
bars = ax12.bar(range(0, len(power), 10), power[::10], width=8, color='#e040fb', alpha=0.8, edgecolor='white', linewidth=0.5)
ax12.set_xlabel('Path Index', color='white', fontsize=10)
ax12.set_ylabel('Power (W)', color='#e040fb', fontsize=10, fontweight='bold')
ax12.set_title('Power Distribution', color='white', fontsize=11, fontweight='bold')
ax12.tick_params(colors='white')
ax12.set_facecolor('#0d1117')
for spine in ax12.spines.values():
    spine.set_color('#444')
ax12.spines['left'].set_color('#e040fb')
ax12.grid(True, alpha=0.2, color='gray', axis='y')

# ============================================
# 子图13: 综合统计 (第四行最右)
# ============================================
ax13 = fig.add_subplot(gs[3, 3], facecolor='#0d1117')
ax13.axis('off')

# 计算统计数据
total_distance = path_length[-1]
avg_velocity = np.mean(velocity)
max_slope = np.max(slope_deg[
    np.clip((path_y/resolution).astype(int), 0, map_size[0]-1),
    np.clip((path_x/resolution).astype(int), 0, map_size[1]-1)
])
total_energy = energy[-1] / 1000
avg_slip = np.mean(slip_ratio) * 100

stats_text = f"""
╔══════════════════════════════════════╗
║     MISSION STATISTICS               ║
╠══════════════════════════════════════╣
║  Total Distance:     {total_distance:6.1f} m        ║
║  Avg Velocity:       {avg_velocity:6.2f} m/s      ║
║  Max Slope:          {max_slope:6.1f} deg         ║
║  Total Energy:       {total_energy:6.1f} kJ       ║
║  Avg Slip Ratio:     {avg_slip:6.1f} %          ║
║  Path Points:        {n_points:6d}            ║
║  Obstacles:          {len(obstacles):6d}            ║
╠══════════════════════════════════════╣
║  SOIL DISTRIBUTION                   ║
║  Regolith:  {np.sum(soil_types==0)/soil_types.size*100:5.1f}%                ║
║  Firm Soil: {np.sum(soil_types==1)/soil_types.size*100:5.1f}%                ║
║  Rock:      {np.sum(soil_types==2)/soil_types.size*100:5.1f}%                ║
╚══════════════════════════════════════╝

  Lunar Rover Navigation System
  Based on Chang'e-6 Data
  
  Environment -> Dynamics -> Planning
"""

ax13.text(0.5, 0.5, stats_text, transform=ax13.transAxes, fontsize=11,
          verticalalignment='center', horizontalalignment='center',
          color='#00e5ff', fontfamily='monospace', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117', edgecolor='#00e5ff', linewidth=2))

# ============================================
# 添加总标题
# ============================================
fig.suptitle('Digital Twin-Based Lunar Rover Navigation System\nComprehensive Simulation Results', 
             color='white', fontsize=20, fontweight='bold', y=0.98)

# 添加副标题
fig.text(0.5, 0.955, 'Environment Modeling | Terramechanics | Path Planning | Dynamics Simulation', 
         color='#888', fontsize=13, ha='center', style='italic')

# ============================================
# 保存高分辨率图像
# ============================================
plt.savefig('output/awesome_thesis_figure_v2.png', dpi=300, bbox_inches='tight', 
            facecolor='#0d1117', edgecolor='none', pad_inches=0.3)
plt.savefig('output/awesome_thesis_figure_v2.pdf', bbox_inches='tight', 
            facecolor='#0d1117', edgecolor='none', pad_inches=0.3)
print("图表已保存到 output/awesome_thesis_figure_v2.png 和 .pdf")

plt.close()
print("完成!")

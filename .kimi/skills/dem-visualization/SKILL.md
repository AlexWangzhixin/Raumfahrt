# DEM 高保真可视化 Skill

用于生成月球/行星表面数字高程模型（DEM）的高保真可视化，适用于博士论文第三章环境建模章节。

## 核心能力

- 生成高保真月球地形DEM（虹湾、南极艾特肯盆地等）
- 2D热力图 + 等高线可视化
- 3D立体视角（表面图+投影）
- 多图拼接（2D+3D上下/左右布局）
- 中文字体自动配置
- 撞击坑识别可视化
- 月面光照渲染效果

## 使用场景

- 图3-3类：地形DEM可视化效果
- 图3-4类：撞击坑识别可视化
- 图3-5类：月面光照渲染效果
- 多源数据融合三维地形重建
- 高分辨率着陆区DEM展示
- 复杂地形（撞击坑、山脉）可视化

## 最佳实践

### 1. 中文字体配置（统一标准）

```python
from matplotlib import font_manager

def configure_chinese_font():
    """配置matplotlib中文字体"""
    candidates = [
        "Microsoft YaHei",
        "SimHei", 
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            # 统一字体大小标准
            plt.rcParams["font.size"] = 16
            plt.rcParams["axes.titlesize"] = 18
            plt.rcParams["axes.labelsize"] = 16
            plt.rcParams["xtick.labelsize"] = 14
            plt.rcParams["ytick.labelsize"] = 14
            return name
    return "DejaVu Sans"
```

### 2. 颜色条配置标准

```python
# 颜色条 - 垂直显示，增大字体，避免重叠
cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.12)
cbar.set_label('高程 (m)', fontsize=14, rotation=90, labelpad=15)
cbar.ax.tick_params(labelsize=12)
```

### 3. 图片布局标准

```python
# 2D图（上半部分）
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
# ... 绘制代码 ...
plt.tight_layout(rect=[0, 0, 1, 0.98])

# 3D图（下半部分）
fig = plt.figure(figsize=(14, 6), dpi=300)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
# ... 绘制代码 ...
plt.tight_layout(rect=[0, 0, 1, 0.98])

# 拼接（不使用总标题）
from PIL import Image
img_2d = Image.open(path_2d)
img_3d = Image.open(path_3d)
combined = Image.new('RGB', (target_width, new_height_2d + new_height_3d), 'white')
combined.paste(img_2d_resized, (0, 0))
combined.paste(img_3d_resized, (0, new_height_2d))
```

### 4. 地形生成参数

#### 虹湾着陆区 (Sinus Iridum)

| 参数 | 值 | 说明 |
|------|-----|------|
| 中心位置 | N44.1°, W31.5° | 嫦娥三号预选着陆区 |
| 直径 | 259 km | 月海海湾 |
| 典型高程 | -2500 m | 月海平原 |
| 特征 | 侏罗山脉、少量撞击坑 | 西北边缘有山 |
| 粗糙度 | low | 相对平坦 |

```python
# 基础地形：月海平原
base_elevation = -2500 + 80 * np.sin(X/8) * np.cos(Y/8)

# 侏罗山脉（西北侧）
mountain_north = 750 * np.exp(-((Y - 46)**2) / 15) * (X < 40)
mountain_west = 550 * np.exp(-((X - 6)**2) / 12) * (Y > 12)

# 撞击坑（碗形凹陷）
crater_mask = dist < radius
crater_elevation[crater_mask] -= depth * (1 - (dist[crater_mask] / radius)**2)
```

#### 南极艾特肯盆地 (SPA Basin)

| 参数 | 值 | 说明 |
|------|-----|------|
| 中心位置 | 53°S, 169°W | 月球背面 |
| 直径 | 2500 km | 月球最大撞击盆地 |
| 深度 | 13 km | 最深点约-9km |
| 特征 | 多环结构、大量撞击坑 | 复杂地形 |
| 粗糙度 | high | 极度崎岖 |

```python
# 多环结构
ring1 = 950 * np.exp(-((dist_from_center - 40)**2) / 40)   # 外环隆起
ring2 = -800 * np.exp(-((dist_from_center - 28)**2) / 32)  # 中环凹陷
ring3 = 600 * np.exp(-((dist_from_center - 14)**2) / 22)   # 内环隆起

# 带中央峰的撞击坑
if has_central_peak:
    central_peak_height = depth * 0.45
    peak_radius = radius * 0.12
    crater_shape = -depth * (1 - (dist / radius)**2)
    crater_shape[peak_mask] += central_peak_height * (1 - (dist[peak_mask] / peak_radius)**2)
```

### 5. 撞击坑识别可视化（图3-4）

```python
# 使用不同颜色标记识别结果
colors = {
    'true_positive': '#00FF00',   # 绿色：正确识别
    'false_negative': '#FF0000',  # 红色：漏检
    'false_positive': '#0000FF',  # 蓝色：误检
}

# 绘制真实撞击坑（绿色圆圈）
for crater in true_craters:
    circle = plt.Circle((crater.x, crater.y), crater.radius, 
                       fill=False, color=colors['true_positive'], linewidth=2)
    ax.add_patch(circle)

# 绘制漏检（红色叉号）
ax.scatter(missed_x, missed_y, c=colors['false_negative'], 
          marker='x', s=100, linewidths=2, label='漏检')

# 绘制误检（蓝色三角形）
ax.scatter(false_x, false_y, c=colors['false_positive'], 
          marker='^', s=100, label='误检')
```

### 6. 光照渲染效果（图3-5）

```python
def apply_sunlight(elevation, sun_azimuth, sun_elevation):
    """
    应用太阳光照效果
    
    Args:
        elevation: 高程数据
        sun_azimuth: 太阳方位角（度）
        sun_elevation: 太阳高度角（度）
    """
    from scipy.ndimage import sobel
    
    # 计算坡度
    dz_dx = sobel(elevation, axis=1)
    dz_dy = sobel(elevation, axis=0)
    
    # 太阳光方向向量
    azimuth_rad = np.radians(sun_azimuth)
    elevation_rad = np.radians(sun_elevation)
    
    sun_x = np.cos(elevation_rad) * np.sin(azimuth_rad)
    sun_y = np.cos(elevation_rad) * np.cos(azimuth_rad)
    sun_z = np.sin(elevation_rad)
    
    # 计算光照强度（Lambert反射模型）
    normal = np.dstack([dz_dx, dz_dy, np.ones_like(elevation)])
    normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)
    
    sun_dir = np.array([sun_x, sun_y, sun_z])
    intensity = np.dot(normal, sun_dir)
    intensity = np.clip(intensity, 0.2, 1.0)  # 环境光底值0.2
    
    return intensity
```

## 输出规范

| 格式 | DPI | 尺寸 | 用途 |
|------|-----|------|------|
| PNG | 300 | 2400px宽 | 论文插入 |
| PDF | - | - | LaTeX矢量图 |
| EPS | - | - | 备用矢量格式 |

## 依赖

```
numpy>=2.0
scipy>=1.13
matplotlib>=3.9
Pillow>=9.0
pyyaml>=6.0
```

## 文件命名规范

```
fig_3_3_dem_2d_final.png      # 图3-3 2D版本
fig_3_3_dem_3d_final.png      # 图3-3 3D版本
fig_3_3_final_combined.png    # 图3-3 拼接版本
fig_3_4_crater_detection.png  # 图3-4 撞击坑识别
fig_3_5_lighting.png          # 图3-5 光照渲染
```

## 常见问题

### Q1: 中文字体显示为方块
**解决**：检查系统是否安装 Microsoft YaHei 或 SimHei

### Q2: 3D图渲染太慢
**解决**：使用降采样 `X[::step, ::step]`，推荐 step=3

### Q3: 颜色条标签重叠
**解决**：增大 `pad` 参数，使用垂直 `rotation=90`

### Q4: 字体大小与正文不一致
**解决**：统一使用 `plt.rcParams["font.size"] = 16`

---

*基于 Raumfahrt 项目图3-3/3-4/3-5 生成经验总结*

# UML 论文架构图生成 Skill

用于生成符合 UML 2.0 规范的博士/硕士论文架构图，特别适用于数字孪生、系统架构等研究领域。

## 核心能力

- 生成 UML 2.0 标准包图（Package Diagram）
- 自动处理中文字体与布局
- 支持多层架构可视化
- 避免元素重叠，确保美观输出

## 使用场景

- 博士/硕士论文系统架构图（图2-1类）
- 数字孪生五维模型架构
- 分层系统架构图（服务层/数据层/实体层）
- UML 组件图与包图

## 最佳实践

### 1. 布局设计原则

| 原则 | 说明 | 经验值 |
|------|------|--------|
| **层间距** | 层与层之间留白充足 | 0.4-0.6 单位 |
| **标题间距** | 层内组件与层标签间距 | 0.2-0.25 单位 |
| **组件间距** | 同级组件间水平间距 | 0.15-0.2 单位 |
| **边距** | 组件与层边框距离 | 0.1-0.15 单位 |
| **文字行距** | 组件内项目行距 | 0.23-0.26 单位 |

### 2. 避免重叠的技巧

```python
# 关键：从底部往上排列参数，确保不超出组件边界
param_y = y + 0.08  # 从底部留边距开始
for param in reversed(params):  # 倒序排列
    ax.text(x + 0.1, param_y, param, ...)
    param_y += 0.11  # 向上递增
```

### 3. 颜色搭配方案

```python
colors = {
    'service': '#e8f5e9',      # 绿色 - 服务层
    'service_border': '#2e7d32',
    'data': '#f3e5f5',          # 紫色 - 数据层
    'data_border': '#6a1b9a',
    'component': '#e3f2fd',     # 蓝色 - 组件/实体层
    'component_border': '#1565c0',
    'note': '#fff9c4',          # 黄色 - 注释
    'note_border': '#f57f17',
}
```

### 4. 层高度设计

| 层类型 | 推荐高度 | 说明 |
|--------|----------|------|
| 简单服务层 | 1.6-1.75 | 3个子项，无参数 |
| 数据层 | 1.6-1.7 | 3个子项 |
| 复杂实体层 | 2.6-2.8 | 4个子项+4个参数 |

### 5. 组件定位公式

```python
# 组件Y位置 = 层Y位置 + 边距
comp_y = layer_y + 0.15  # 标准边距

# 对于需要下移的层（避免嵌入标题）
comp_y = layer_y + 0.22  # 增大边距
```

## 常见问题与解决

### 问题1：文字与底部边框重叠
**原因**：文字从顶部往下排列，底部参数无足够空间
**解决**：参数从底部往上倒序排列，预留边距

### 问题2：组件嵌入层标题标签
**原因**：组件Y位置过高，与标签区域重叠
**解决**：增大 `comp_y = layer_y + offset` 中的 offset 值

### 问题3：右侧组件过于贴近边框
**原因**：组件宽度 + X位置接近层宽度
**解决**：减小组件宽度或左移X位置

### 问题4：注释框与组件重叠
**原因**：注释Y位置过高
**解决**：将注释位置下调到组件中部或底部

## 代码模板

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(8.27, 9.5), dpi=200)
ax.set_xlim(0, 8.27)
ax.set_ylim(0, 9.5)
ax.axis('off')

# 绘制UML包（层）
def draw_package(ax, x, y, width, height, name, color_key, border_color):
    tab_width = min(len(name) * 0.28 + 0.3, width * 0.85)
    # 标签页
    tab = FancyBboxPatch((x, y + height - 0.42), tab_width, 0.42,
                         boxstyle="round,pad=0,rounding_size=0.05",
                         facecolor=border_color,
                         edgecolor='#333',
                         linewidth=1.5)
    ax.add_patch(tab)
    ax.text(x + tab_width/2, y + height - 0.21, name, 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    # 主体
    body = FancyBboxPatch((x, y), width, height - 0.42,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          facecolor=color_key,
                          edgecolor='#333',
                          linewidth=1.5)
    ax.add_patch(body)

# 绘制组件
def draw_component(ax, x, y, width, height, name, items, color_key, border_color):
    comp = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0,rounding_size=0.03",
                          facecolor=color_key,
                          edgecolor=border_color,
                          linewidth=1.5)
    ax.add_patch(comp)
    
    # 标题
    ax.text(x + width/2, y + height - 0.22, name, 
            ha='center', va='center', fontsize=9, fontweight='bold', color=border_color)
    
    # 分隔线
    ax.plot([x + 0.08, x + width - 0.08], 
            [y + height - 0.38, y + height - 0.38], 
            color=border_color, linewidth=0.8)
    
    # 内容项
    item_y = y + height - 0.52
    for item in items:
        ax.text(x + 0.1, item_y, item, ha='left', va='center',
                fontsize=8, color='#333')
        item_y -= 0.25

# 保存
plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.1)
```

## 版本迭代记录

| 版本 | 改进点 |
|------|--------|
| v1-v3 | 基础布局 |
| v4-v5 | 修复重叠问题 |
| v6-v8 | 移除<<部件>>标签，增加参数显示 |
| v9-v10 | 优化嵌套关系 |
| v11-v13 | 下移组件避免嵌入标题，最终美化 |

## 输出规范

- **格式**: PNG (300 DPI) + PDF (矢量)
- **尺寸**: A4 竖版 (8.27 x 11.69 英寸)
- **边距**: 0.1 英寸 tight
- **背景**: 纯白

## 依赖

```
matplotlib>=3.9
numpy>=2.0
```

---

*基于 Raumfahrt 项目图2-1 生成经验总结*

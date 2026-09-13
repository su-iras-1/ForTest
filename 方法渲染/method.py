import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 1. 创建坐标网格 (A, B)
a = np.linspace(-60, 60, 300)
b = np.linspace(-60, 60, 300)
A, B = np.meshgrid(a, b)

# 2. 模拟数据场 Z (WATD in Mbps)
# 这里使用双高斯分布模拟图中两个聚集区域
Z1 = 300 * np.exp(-((A - 20)**2 + (B - 15)**2) / 400)
Z2 = 280 * np.exp(-((A - 10)**2 + (B + 25)**2) / 350)
Z = np.maximum(Z1, Z2)

# 3. 创建画布
fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)

# 4. 绘制填充等高图 (使用 parula 或 viridis 类似配色)
levels = np.linspace(0, 300, 13)
cf = ax.contourf(A, B, Z, levels=levels, cmap='viridis', extend='both')

# 5. 绘制黑色等高线并标出数值
cs = ax.contour(A, B, Z, levels=[50, 133, 200], colors='black', linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

# 6. 绘制红色边界圆 (半径 R=50)
circle = Circle((0, 0), radius=50, color='red', fill=False, linewidth=1.5)
ax.add_patch(circle)

# 7. 叠加散点数据 (筛选 Z > 50 的区域生成散点)
mask = Z > 40
scatter_a = A[mask][::20]  # 适当下采样
scatter_b = B[mask][::20]
ax.scatter(scatter_a, scatter_b, color='blue', edgecolors='cyan', s=18, linewidths=0.5, zorder=3)

# 8. 颜色条 (Colorbar)
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.ax.set_title('WATD in Mbps', fontsize=9, pad=8)

# 9. 坐标轴与细节调整
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_xticks([-50, 0, 50])
ax.set_yticks([-50, 0, 50])
ax.set_xlabel('A (degree)', fontsize=11)
ax.set_ylabel('B (degree)', fontsize=11)
ax.set_aspect('equal') # 确保圆形不变形

plt.tight_layout()
plt.show()
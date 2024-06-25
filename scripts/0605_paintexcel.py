import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 读取Excel文件
df = pd.read_excel('/home/ps/huichenchen/mmdetection3d/results2/analysis2/0527-parallel-withddcm.xlsx')

# 设置图形大小
fig, axes = plt.subplots(4, 5, figsize=(20, 10))
axes = axes.flatten()

# 使用Excel表的前20列
categories = df.columns[:20]

# 定义颜色调色板，为 miou 使用深红色，其他类别使用不同的颜色
color_palette = ['#6496f5', '#64e6f5', '#1e3c96', '#ffbb78', '#501eb4', '#6450fa', '#9b1e1e', '#ff28c8', 
                 '#961e5a', '#ff00ff', '#ff96ff', '#4b004b', '#af004b', '#ffc800', '#ff7832', '#00af00', 
                 '#873c00', '#96f050', '#fff096', '#ff0000']

# 绘制每个类别的IoU变化
for idx, (category, color) in enumerate(zip(categories, color_palette)):
    ax = axes[idx]
    ax.plot(df.index.to_numpy() + 1, df[category], label=category, color=color, linewidth=1.5)
    

    
    # 设置子图标题
    ax.set_title(category, fontsize='25')
    
    # 设置坐标轴
    ax.set_xlim(1, 36)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 18, 36])
    ax.set_yticks([0, 0.5, 1])

    ax.tick_params(axis='x', labelsize=25)  # 设置x轴标签文字大小为12
    ax.tick_params(axis='y', labelsize=25)  # 设置y轴标签文字大小为12
    
    # 隐藏除最左边和最下边的坐标轴标签
    if idx % 5 != 0:
        ax.set_yticklabels([])
    if idx < 15:
        ax.set_xticklabels([])

# 设置总体标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei
plt.suptitle('各类别IoU随训练周期的变化', fontsize='25')

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 1])

# 保存图像
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/analysis_paint4/0605-2-parallel-withddcm.png')

# 显示图像
# plt.show()
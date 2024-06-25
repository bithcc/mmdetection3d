import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 读取Excel文件
df = pd.read_excel('/home/ps/huichenchen/mmdetection3d/results2/analysis2/0527-onlylaser.xlsx')

# 设置图形大小并调整布局
plt.figure(figsize=(20, 12))
plt.subplots_adjust(right=0.8)  # 调整子图参数，为图例留出空间

# 使用Excel表的前20列
categories = df.columns[:20]

# 定义一组颜色，为 miou 使用深红色，其他类别使用不同的颜色
color_palette = ['#6496f5', '#64e6f5', '#1e3c96', '#ffbb78', '#501eb4', '#6450fa', '#9b1e1e', '#ff28c8', 
                 '#961e5a', '#ff00ff', '#ff96ff', '#4b004b', '#af004b', '#ffc800', '#ff7832', '#00af00', 
                 '#873c00', '#96f050', '#fff096', '#ff0000']

# color_palette = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#98FF98', '#87CEEB', 
#                  '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', 
#                  '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# miou为深红色，其他类别使用提供的颜色
colors = color_palette[:-1] + ['#000080']  # 最后一个颜色为miou
linewidths = [1.5] * 19 + [3.5]  # 默认1.5，miou为2.5

# 为每个类别绘图
for category, color, lw in zip(categories, colors, linewidths):
    plt.plot(df.index.to_numpy() + 1, df[category], label=category, color=color, linewidth=lw)

plt.rcParams['font.sans-serif'] = ['SimHei']#指定字体为SimHei
# 显示图例，一列显示，更大的文字，调整图例位置
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize='25')

# 设置标题和标签
plt.title('各类别IoU随训练周期的变化', fontsize='30')
plt.xlabel('训练周期', fontsize='30')
plt.ylabel('IoU', fontsize='30')

# 设置坐标轴标签大小
plt.xticks(range(1, 37,2), fontsize='30')  # Ensure x-ticks are from 1 to 36
plt.yticks(fontsize='30')

# 设置x轴和y轴的起始点为1，并让它们在原点相交
plt.xlim(1, 36)
plt.ylim(bottom=0)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(1, color='black', linewidth=0.8)

# 显示图形
# plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/analysis_paint/0422-base3d.png')
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/analysis_paint3/0527-onlylaser.png')

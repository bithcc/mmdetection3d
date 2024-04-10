import matplotlib.pyplot as plt
import numpy as np

# 类别及其对应的RGB颜色
categories_colors = {
    'car': (100, 150, 245),  
    'bicycle': (100, 230, 245),  
    'motorcycle': (30, 60, 150),
    'truck': (80, 30, 180),  
    'other-vehicle': (100, 80, 250),  
    'person': (155, 30, 30),
    'bicyclist': (255, 40, 200),  
    'motorcyclist': (150, 30, 90),  
    'road': (255, 0, 255),
    'parking': (255, 150, 255),  
    'sidewalk': (75, 0, 75),  
    'other-ground': (175, 0, 75),
    'building': (255, 200, 0),  
    'fence': (255, 120, 50),  
    'vegetation': (0, 175, 0),
    'trunk': (135, 60, 0),  
    'terrain': (150, 240, 80),  
    'pole': (255, 240, 150),
    'traffic-sign': (255, 0, 0),  
    'ignore': (255,255,255),
}

# 计算需要的行数和列数
n = len(categories_colors)  # 类别总数
cols = 5  # 假设我们希望每行显示3个图例
rows = int(np.ceil(n / cols))  # 计算需要的行数

# 创建一个足够大的画布
plt.figure(figsize=(cols * 2, rows * 2))

for i, (category, rgb) in enumerate(categories_colors.items(), start=1):
    ax = plt.subplot(rows, cols, i)
    ax.set_facecolor(np.array(rgb) / 255.0)  # 设置子图背景颜色
    ax.set_xticks([])  # 移除x轴刻度
    ax.set_yticks([])  # 移除y轴刻度
    ax.set_title(category)  # 设置子图标题为类别名称

plt.tight_layout()  # 调整子图间距
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0408_colormap.png')

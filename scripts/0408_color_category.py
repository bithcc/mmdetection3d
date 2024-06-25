import matplotlib.pyplot as plt
import numpy as np

# 类别及其对应的RGB颜色
categories_colors = {
    '0:car': (100, 150, 245),  
    '1:bicycle': (100, 230, 245),  
    '2:motorcycle': (30, 60, 150),
    '3:truck': (80, 30, 180),  
    '4:other-vehicle': (100, 80, 250),  
    '5:person': (155, 30, 30),
    '6:bicyclist': (255, 40, 200),  
    '7:motorcyclist': (150, 30, 90),  
    '8:road': (255, 0, 255),
    '9:parking': (255, 150, 255),  
    '10:sidewalk': (75, 0, 75),  
    '11:other-ground': (175, 0, 75),
    '12:building': (255, 200, 0),  
    '13:fence': (255, 120, 50),  
    '14:vegetation': (0, 175, 0),
    '15:trunk': (135, 60, 0),  
    '16:terrain': (150, 240, 80),  
    '17:pole': (255, 240, 150),
    '18:traffic-sign': (255, 0, 0),  
    '19:ignore': (255,255,255),
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
    ax.set_title(category,size=18)  # 设置子图标题为类别名称

plt.tight_layout()  # 调整子图间距
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0522_colormap_test.png')

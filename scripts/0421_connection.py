import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# 绘制网络层
layer_x = [1, 3, 5, 7, 9]  # X坐标位置
layer_text = ['Input Layer', 'Layer 1', 'Layer 2', 'Layer 3', 'Output Layer']

for x, text in zip(layer_x, layer_text):
    ax.text(x, 2.5, text, fontsize=12, ha='center')
    ax.scatter(x, 2.5, s=1000, c='skyblue', edgecolor='black', alpha=0.6, linewidth=2)

# 绘制连接
for i in range(len(layer_x)-1):
    con = ConnectionPatch(xyA=(layer_x[i], 2.5), xyB=(layer_x[i+1], 2.5), coordsA='data', coordsB='data',
                          arrowstyle='->', shrinkB=5)
    ax.add_artist(con)

# 绘制残差连接
res_con = ConnectionPatch(xyA=(layer_x[1], 2.5), xyB=(layer_x[3], 2.5), coordsA='data', coordsB='data',
                          connectionstyle="arc3,rad=.5", arrowstyle='->', linestyle='--', color='red', shrinkB=5)
ax.add_artist(res_con)

# 标注残差连接
ax.text((layer_x[1]+layer_x[3])/2, 3.8, 'Residual Connection', color='red', fontsize=12, ha='center')

plt.title('Illustration of Residual Connection')
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0421_connection.png')

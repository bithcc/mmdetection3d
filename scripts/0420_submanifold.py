import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成随机点云数据
x = np.random.uniform(low=-3, high=3, size=100)
y = np.random.uniform(low=-3, high=3, size=100)
z = np.random.uniform(low=-3, high=3, size=100)
ax.scatter(x, y, z)

# 高亮一个子区域，模拟卷积核
kernel_size = 10  # 假设卷积核大小为10x10x10
kernel_center = np.mean([x, y, z], axis=1)
ax.scatter(kernel_center[0], kernel_center[1], kernel_center[2], color='r', s=100)

# 添加网格，表示体素化
ax.set_xticks(np.arange(-3, 4, 1))
ax.set_yticks(np.arange(-3, 4, 1))
ax.set_zticks(np.arange(-3, 4, 1))
ax.grid(which='major', color='#CCCCCC', linestyle='--')

# plt.title('Submanifold 3D Convolution Example')
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/submanifold2.png')

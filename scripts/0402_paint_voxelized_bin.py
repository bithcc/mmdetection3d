#用于查看bin格式原始点云
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

bin_file = '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/08/velodyne/000000.bin'
ply_file = '/home/ps/huichenchen/mmdetection3d/results2/000000t.ply'

points = np.fromfile(bin_file,dtype=np.float32).reshape(-1,4)[:,:-1]

pcd = o3d.geometry.PointCloud()

points=points.astype(np.float64)

# print(points.shape)
# print(np.any(np.isnan(points)),np.any(np.isinf(points)))

pcd.points = o3d.utility.Vector3dVector(points)


# 将点云从笛卡尔坐标系转换到柱面坐标系
rho = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
phi = np.arctan2(points[:, 1], points[:, 0])
z = points[:, 2]

# 定义柱面体素的分割参数
rho_bins = np.linspace(rho.min(), rho.max(), num=11)  # 将径向距离分割成10段
phi_bins = np.linspace(-np.pi, np.pi, num=11)  # 将角度分割成20段
z_bins = np.linspace(z.min(), z.max(), num=2)  # 将高度分割成1段

# 对每个点计算所属的体素索引
rho_indices = np.digitize(rho, rho_bins) - 1
phi_indices = np.digitize(phi, phi_bins) - 1
z_indices = np.digitize(z, z_bins) - 1

# 可视化：为了简化，我们只根据体素索引改变点云的颜色
# voxel_indices = rho_indices + phi_indices * len(rho_bins) + z_indices * len(rho_bins) * len(phi_bins)
voxel_indices = rho_indices + phi_indices + z_indices
colors = plt.get_cmap("hsv")(voxel_indices / voxel_indices.max())[:, :3]
# print(voxel_indices)
# print(colors)

# 更新点云颜色并显示
pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud(ply_file,pcd)


import open3d as o3d
import numpy as np
# 加载PCD文件
pcd_path = '/home/ps/huichenchen/mmdetection3d/results2/exp/robosense128-road/ruby_ruby002_baoshenlu_1200303103447_5.pcd'
pcd = o3d.io.read_point_cloud(pcd_path, print_progress=True)

# 打印出PCD文件的结构信息
print("Point cloud has:")
print(f" - {len(pcd.points)} points")

if pcd.has_colors():
    print(" - Colors per point")
if pcd.has_normals():
    print(" - Normals per point")

# 查看额外的属性（如果有）
if hasattr(pcd, 'attributes'):
    print(" - Additional attributes:")
    for attr in pcd.attributes:
        print(f"   - {attr}: {pcd.attributes[attr].shape}")

# 打印前几个点的数据（点坐标）
print("\nSample data (first 5 points):")
print(np.asarray(pcd.points)[:5])

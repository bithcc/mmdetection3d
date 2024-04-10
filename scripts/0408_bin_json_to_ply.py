import numpy as np
import json
import open3d as o3d

# /home/ps/huichenchen/mmdetection3d/results2/test2/preds/1659079941.794228403.json
# /home/ps/huichenchen/mmdetection3d/results2/test2/preds/1659079941.894149173.json
# /home/ps/huichenchen/mmdetection3d/results2/test2/preds/1659079941.989026227.json
# 替换成你的文件路径
bin_file_path = '/home/ps/huichenchen/mmdetection3d/results2/rs16/test/1615876637.188017000.bin'
json_file_path = '/home/ps/huichenchen/mmdetection3d/results2/rs16/test/preds/1615876637.188017000.json'
ply_file_path = '/home/ps/huichenchen/mmdetection3d/results2/rs16/test/1615876637.188017000.ply'

# 读取点云数据
point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)[:,:-1]
point_cloud = point_cloud.astype(np.float64)

# 读取标签数据
with open(json_file_path, 'r') as f:
    labels = json.load(f)['pts_semantic_mask']

# 假设有4个标签，分配不同的颜色 (红, 绿, 蓝, 黄)
colors = np.array([
    [100, 150, 245],  #car 
    [100, 230, 245], #bicycle  
    [30, 60, 150],   #motorcycle
    [80, 30, 180], #truck
    [100,80,250], #other-vehicle
    [155,30,30], #person
    [255,40,200], #bicyclist
    [150,30,90], #motorcyclist
    [255,0,255], #road
    [255,150,255], #parking
    [75,0,75], #sidewalk
    [175,0,75], #other-ground
    [255,200,0], #building
    [255,120,50], #fence
    [0,175,0], #vegetation
    [135,60,0], #trunk
    [150,240,80], #terrain
    [255,240,150], #pole
    [255,0,0], #traffic-sign
])

# 根据标签为点云分配颜色
point_colors = np.array([colors[label] for label in labels])

# 创建Open3D点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)  # 颜色值需归一化到[0, 1]

# 保存为PLY文件
o3d.io.write_point_cloud(ply_file_path, pcd)

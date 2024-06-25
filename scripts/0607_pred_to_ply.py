import numpy as np
import json
import open3d as o3d

# 第一步：读取点云数据和标签，生成带有颜色信息的ply文件
bin_file_path = '/mnt/datasets/huichenchen/robosense/rs128/bin/ruby119_longzhudadao_1200423181920_2595.bin'
json_file_path = '/mnt/datasets/huichenchen/robosense/rs128/multi-json/preds/ruby119_longzhudadao_1200423181920_2595.json'
bin_file_path_with_labels = '/mnt/datasets/huichenchen/robosense/rs128/label_bin/ruby119_longzhudadao_1200423181920_2595_labels.bin'  # 包含点云坐标和标签的bin文件
ply_file_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/multi-ply/ruby119_longzhudadao_1200423181920_2595.ply'
filtered_ply_file_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/multi-filter-ply/ruby119_longzhudadao_1200423181920_2595-filtered2.ply'


# 读取点云数据
point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)[:, :-1]
point_cloud = point_cloud.astype(np.float64)

# 读取标签数据
with open(json_file_path, 'r') as f:
    labels = json.load(f)['pts_semantic_mask']

# 分配颜色
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
point_colors = np.array([colors[label % len(colors)] for label in labels])

# 创建Open3D点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)  # 颜色值需归一化到[0, 1]

# 保存为PLY文件
o3d.io.write_point_cloud(ply_file_path, pcd)

# 第二步：读取另一个bin文件，筛选出类别不为19的点云坐标，并过滤PLY文件


# 读取包含标签的点云数据
point_cloud_with_labels = np.fromfile(bin_file_path_with_labels, dtype=np.float32).reshape(-1, 4)
labels_with_labels = point_cloud_with_labels[:, 3].astype(int)  # 假设标签是第四列

# 筛选出类别不为19的点云坐标
mask = (labels_with_labels != 19)
non_background_indices = np.where(mask)[0]

# 读取PLY文件
filtered_pcd = o3d.io.read_point_cloud(ply_file_path)

# 过滤点云，仅保留与筛选出的坐标相对应的点
filtered_points = point_cloud[non_background_indices]
filtered_colors = point_colors[non_background_indices]

# 更新Open3D点云对象
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)

# 保存过滤后的PLY文件

o3d.io.write_point_cloud(filtered_ply_file_path, filtered_pcd)

print(f"Filtered PLY file with color information has been saved to {filtered_ply_file_path}")
# # 示例使用
# annotation_bin_path = '/mnt/datasets/huichenchen/robosense/rs32/test/32_yueanerlu_1190711151651_990_labels.bin'  # 替换为实际的文件路径
# prediction_json_path = '/mnt/datasets/huichenchen/robosense/rs32/test/32_yueanerlu_1190711151651_990.json'  # 替换为实际的文件路径
# output_ply_path = '/mnt/datasets/huichenchen/robosense/rs32/test/0607_test.ply'  # 替换为希望保存的ply文件路径

# process_and_save_ply(annotation_bin_path, prediction_json_path, output_ply_path)

# bin_path_points = '/mnt/datasets/huichenchen/robosense/rs32/test/32_yueanerlu_1190711151651_990_labels.bin'  # 第一个bin文件，包含点云坐标和标签
# bin_path_full = '/mnt/datasets/huichenchen/robosense/rs32/bin/32_yueanerlu_1190711151651_990.bin'      # 第二个bin文件，包含所有点云数据
# json_path = '/mnt/datasets/huichenchen/robosense/rs32/test/32_yueanerlu_1190711151651_990.json'            # json文件，包含预测标签
# output_ply_path = '/mnt/datasets/huichenchen/robosense/rs32/test/0607_test.ply'            # 输出的ply文件路径
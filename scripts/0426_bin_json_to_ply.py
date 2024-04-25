import os
import numpy as np
import json
import open3d as o3d

def process_files(bin_dir, json_dir, ply_dir):
    # 遍历bin文件夹中的所有bin文件
    for filename in os.listdir(bin_dir):
        if filename.endswith('.bin'):
            base_name = os.path.splitext(filename)[0]
            
            bin_file_path = os.path.join(bin_dir, filename)
            json_file_path = os.path.join(json_dir, f"{base_name}.json")
            ply_file_path = os.path.join(ply_dir, f"{base_name}.ply")
            
            # 读取点云数据
            point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)[:,:-1]
            point_cloud = point_cloud.astype(np.float64)

            # 读取标签数据
            with open(json_file_path, 'r') as f:
                labels = json.load(f)['pts_semantic_mask']

            # 定义颜色数组
            colors = np.array([
                [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180], [100,80,250], 
                [155,30,30], [255,40,200], [150,30,90], [255,0,255], [255,150,255], [75,0,75], 
                [175,0,75], [255,200,0], [255,120,50], [0,175,0], [135,60,0], [150,240,80], 
                [255,240,150], [255,0,0]
            ])

            # 根据标签为点云分配颜色
            point_colors = np.array([colors[label] for label in labels])

            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)  # 颜色值需归一化到[0, 1]

            # 保存为PLY文件
            o3d.io.write_point_cloud(ply_file_path, pcd)
            print(f"Processed and saved: {ply_file_path}")

# 指定三个文件夹的路径
bin_dir = '/home/ps/huichenchen/mmdetection3d/results2/exp/bin_output/bin_robosense32-park'
json_dir = '/home/ps/huichenchen/mmdetection3d/results2/exp/json_output/json_robosense32-park/preds'
ply_dir = '/home/ps/huichenchen/mmdetection3d/results2/exp/ply_output/ply_robosense32-park'

# 调用函数
process_files(bin_dir, json_dir, ply_dir)

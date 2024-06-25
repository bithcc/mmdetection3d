import numpy as np
import os
from plyfile import PlyData, PlyElement

# 标签与颜色的对应关系字典
label_color_map = {
    0: (100, 150, 245),    # 例如：'vehicle' 对应红色
    1: (100, 230, 245),    # 'bicycle' 对应绿色
    2: (30, 60, 150),    # 'motorcycle' 对应蓝色
    3: (80, 30, 180), 
    4: (100, 80, 250),  
    5: (150, 30, 30), 
    6: (255, 40, 200),  
    7: (150, 30, 90),  
    8: (255, 0, 255),
    9: (255, 150, 255),  
    10: (75, 0, 75),  
    11: (175, 0, 75),
    12: (255, 200, 0),  
    13: (255, 120, 50),  
    14: (0, 175, 0),
    15: (135, 60, 0),  
    16: (150, 240, 80),  
    17: (255, 240, 150),
    18: (255, 0, 0) , # 'unknown' 对应灰色
    19: (255, 255, 255), 
}




def load_bin_file(bin_file_path):
    # 读取二进制文件，假设前3个float32是坐标，第4个float32是反射率
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return points

def assign_colors(points, labels):
    # 为每个点根据标签分配颜色
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        color = label_color_map.get(label, (255, 255, 255))  # 默认为灰色
        colors[i] = color
    return colors

def save_as_ply(points, colors, output_path):
    # 将点云和颜色信息保存为PLY文件
    vertex = np.array([
        (point[0], point[1], point[2], color[0], color[1], color[2])
        for point, color in zip(points, colors)
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_path)

# 指定文件路径
pcd_bin_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/bin/ruby119_longzhudadao_1200423181920_2595.bin'  # 点云文件路径
label_bin_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/label_bin/ruby119_longzhudadao_1200423181920_2595_labels.bin'      # 标注文件路径
output_ply_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/label_ply/ruby119_longzhudadao_1200423181920_2595.ply'      # 输出的PLY文件路径

# 读取点云和标注
point_cloud = load_bin_file(pcd_bin_path)
labels = load_bin_file(label_bin_path)[:, 3]  # 假设标签存储在第四列

# 为点云分配颜色
colors = assign_colors(point_cloud, labels)

# 保存为PLY文件
save_as_ply(point_cloud, colors, output_ply_path)
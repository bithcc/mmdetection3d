import numpy as np
import open3d as o3d

color_map = {
    0: (100, 150, 245),  
    1: (100, 230, 245),  
    2: (30, 60, 150),
    3: (80, 30, 180),  
    4: (100, 80, 250),  
    5: (155, 30, 30),
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
    18: (255, 0, 0),  
    19: (255,255,255),
}
labels_map = dict({
    0: 19,
    1: 19,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 19,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 19
})

def read_bin_point_cloud(bin_file):
    """读取.bin格式的点云数据"""
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)  # XYZ + 反射率
    return point_cloud[:, :3]  # 只返回XYZ坐标

def read_label_file(label_file):
    """读取.label格式的标签数据"""
    labels = np.fromfile(label_file, dtype=np.uint32).reshape(-1)
    semantic_labels = labels & 0xFFFF  # 获取语义标签
    return semantic_labels

def apply_label_map(labels, labels_map):
    """应用标签映射转换标签"""
    mapped_labels = np.array([labels_map[label] if label in labels_map else label for label in labels])
    return mapped_labels

def assign_colors_to_labels(labels, color_map):
    """根据标签为点云分配颜色"""
    colors = np.array([color_map[label] if label in color_map else (255, 255, 255) for label in labels])
    return colors

bin_file = "/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/08/velodyne/000000.bin"
label_file = "/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/08/labels/000000.label"
output_ply_file = "/home/ps/huichenchen/mmdetection3d/results2/test/output.ply"

# 假设你已经有了点云数据point_cloud，以及color_map和labels_map
# 读取并应用标签映射
point_cloud = read_bin_point_cloud(bin_file)
labels = read_label_file(label_file)
mapped_labels = apply_label_map(labels, labels_map)

# 为转换后的标签分配颜色
colors = assign_colors_to_labels(mapped_labels, color_map)


# 创建并保存带有颜色的点云为PLY文件
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # 假设point_cloud是从.bin文件读取的
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 颜色值需归一化到[0, 1]
o3d.io.write_point_cloud(output_ply_file, pcd)

import numpy as np
import json
from tqdm import tqdm
import math
import os

# 语义分割标签映射字典
semantic_label_map = {
    'vehicle': 0,
    'big_vehicle': 3,
    'huge_vehicle': 3,
    'pedestrian': 5,
    'bicycle': 1,
    'motorcycle': 2,
    'tricycle': 2,
    'cone': 18,
    'unknown': 19
}

def euler_to_rotation_matrix(roll, pitch, yaw):
    """计算欧拉角到旋转矩阵的转换。"""
    R_x = np.array([[1, 0, 0],
                     [0, math.cos(roll), -math.sin(roll)],
                     [0, math.sin(roll), math.cos(roll)]])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def is_inside_box(point, box, rotation):
    """判断点是否在旋转框内。"""
    px, py, pz = point[:3]
    bx, by, bz, bw, bh, bl = box
    roll, pitch, yaw = rotation

    rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
    point_relative = np.dot(rotation_matrix.T, np.array([px - bx, py - by, pz - bz]))

    x_min, x_max = -bw / 2, bw / 2
    y_min, y_max = -bh / 2, bh / 2
    z_min, z_max = -bl / 2, bl / 2
    return (x_min <= point_relative[0] <= x_max) and \
           (y_min <= point_relative[1] <= y_max) and \
           (z_min <= point_relative[2] <= z_max)

def load_bin(bin_path):
    """从二进制文件中加载点云数据。"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def load_annotations(json_path):
    """从JSON文件中加载标注。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations['labels']

# def assign_semantic_labels(points, annotations):
#     """为点云中的每个点分配语义分割标签。"""
#     labels = np.zeros((points.shape[0],), dtype=np.uint8)
#     for annotation in tqdm(annotations, desc="Assigning labels"):
#         label = semantic_label_map.get(annotation['type'], 19)  # 默认标签为18
#         box = (annotation['center']['x'], annotation['center']['y'], annotation['center']['z'],
#                annotation['size']['x'], annotation['size']['y'], annotation['size']['z'])
#         rotation = (annotation['rotation']['pitch'], annotation['rotation']['roll'], annotation['rotation']['yaw'])
#         for i in range(len(points)):
#             if is_inside_box(points[i], box, rotation):
#                 labels[i] = label
#     return labels

def assign_semantic_labels(points, annotations):
    """为点云中的每个点分配语义分割标签。"""
    labels = np.full((points.shape[0],), -1, dtype=np.int8)  # 初始化标签数组，使用-1表示未分配的点
    for annotation in tqdm(annotations, desc="Assigning labels"):
        label = semantic_label_map.get(annotation['type'], 19)  # 使用字典获取标签，若无则默认为19
        box = (annotation['center']['x'], annotation['center']['y'], annotation['center']['z'],
               annotation['size']['x'], annotation['size']['y'], annotation['size']['z'])
        rotation = (annotation['rotation']['pitch'], annotation['rotation']['roll'], annotation['rotation']['yaw'])
        for i in range(len(points)):
            if is_inside_box(points[i], box, rotation):
                labels[i] = label  # 将点的标签设置为对应的语义标签

    # 将所有未分配的点设置为'unknown'
    labels[labels == -1] = semantic_label_map['unknown']

    return labels

def save_labels_as_bin(points, labels, output_path):
    """将带有标签的点云数据保存到二进制文件。"""
    with open(output_path, 'wb') as f:
        for i in range(points.shape[0]):
            f.write(np.array([points[i][0], points[i][1], points[i][2], labels[i]], dtype=np.float32).tobytes())

# 指定文件路径
bin_path = '/mnt/datasets/huichenchen/robosense/rs128/test/bin/ruby119_nanshandadao_1200421163451_5600.bin'  # 替换为您的bin文件路径
json_path = '/mnt/datasets/huichenchen/robosense/rs128/label/ruby119_nanshandadao_1200421163451_5600.json'  # 替换为您的json文件路径
output_label_path = '/mnt/datasets/huichenchen/robosense/rs128/test/label_bin/ruby119_nanshandadao_1200421163451_5600_2.bin'  # 替换为您想要保存标签的文件路径

# 加载点云数据和注释
points = load_bin(bin_path)
annotations = load_annotations(json_path)

# 分配语义分割标签
labels = assign_semantic_labels(points, annotations)

# 保存标签到bin文件
save_labels_as_bin(points, labels, output_label_path)
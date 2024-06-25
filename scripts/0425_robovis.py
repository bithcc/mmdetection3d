import numpy as np
import json
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import math

# 颜色映射字典128线
color_map = {
    'vehicle': (100, 150, 245),     #0
    'big_vehicle': (80, 30, 180),   #3
    'huge_vehicle': (80, 30, 180),  #3
    'pedestrian': (155, 30, 30),    #5
    'bicycle': (100, 230, 245),     #1
    'motorcycle': (30, 60, 150),    #2
    'tricycle': (30, 60, 150),      #2
    'cone': (255,0,0 ),             #18
    'unknown': (128, 128, 128)      #19
}

# # 颜色映射字典32线
# color_map = {
#     'vehicle': (100, 150, 245),   #car   
#     'big_vehicle': (80, 30, 180),  # 绿色
#     'huge_vehicle': (80, 30, 180), # 蓝色
#     'pedestrian': (155, 30, 30), # 黄色
#     'bicycle': (100, 230, 245),    # 紫色
#     'unknown': (128, 128, 128)   # 灰色
# }




def euler_to_rotation_matrix(roll, pitch, yaw):
    # 计算旋转矩阵
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
    px, py, pz = point[:3]
    bx, by, bz, bw, bh, bl = box
    roll, pitch, yaw = rotation
    
    # 获取旋转矩阵并计算点在框本地坐标系中的位置
    rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
    point_relative = np.dot(rotation_matrix.T, np.array([px-bx, py-by, pz-bz]))
    
    # 检查转换后的点是否在框内
    x_min, x_max = -bw / 2, bw / 2
    y_min, y_max = -bh / 2, bh / 2
    z_min, z_max = -bl / 2, bl / 2
    return (x_min <= point_relative[0] <= x_max) and \
           (y_min <= point_relative[1] <= y_max) and \
           (z_min <= point_relative[2] <= z_max)

def load_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def load_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations['labels']

def color_points(points, annotations):
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    for annotation in tqdm(annotations, desc="Processing annotations"):
        box = (annotation['center']['x'], annotation['center']['y'], annotation['center']['z'],
               annotation['size']['x'], annotation['size']['y'], annotation['size']['z'])
        rotation = (annotation['rotation']['roll'], annotation['rotation']['pitch'], annotation['rotation']['yaw'])
        color = color_map[annotation['type']]
        for i in range(len(points)):
            if is_inside_box(points[i], box, rotation):
                colors[i] = color
    return colors

def save_as_ply(points, colors, output_path):
    vertex = np.array([(point[0], point[1], point[2], color[0], color[1], color[2])
                       for point, color in zip(points, colors)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_path)

bin_path = '/home/ps/huichenchen/mmdetection3d/results2/exp/bin_output/bin_robosense32-road/32_32B413_beierlu_1191010105755_115.bin'
json_path = '/home/ps/huichenchen/mmdetection3d/results2/exp/json_input/json_robosense32-road/32_32B413_beierlu_1191010105755_115.json'
# ply_path = '/home/ps/huichenchen/mmdetection3d/results2/exp/vis_output/vis_robosense128-road-new/ruby_ruby002_baoshenlu_1200303103447_5.ply'
ply_path = '/home/ps/huichenchen/mmdetection3d/results2/0525_32_32B413_beierlu_1191010105755_115.ply'
points = load_bin(bin_path)
annotations = load_annotations(json_path)
colors = color_points(points, annotations)
save_as_ply(points, colors, ply_path)












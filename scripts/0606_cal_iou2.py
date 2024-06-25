#分类别的iou

import numpy as np
import json

# 读取 bin 文件
def read_bin(file_path):
    # 假设点云数据是 float32 类型
    data = np.fromfile(file_path, dtype=np.float32)
    # 每个点包含4个值 (x, y, z, label)
    data = data.reshape(-1, 4)
    return data

# 读取 json 文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['pts_semantic_mask']

# 筛选标签不为 19 的点云
def filter_points(data):
    return data[data[:, 3] != 19]

# 计算 IoU 指标
def calculate_iou(ground_truth, predictions, num_classes):
    ious = []
    for class_id in range(num_classes):
        gt = (ground_truth == class_id)
        pred = (predictions == class_id)
        intersection = np.sum(np.logical_and(gt, pred))
        union = np.sum(np.logical_or(gt, pred))
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

# 读取数据
bin_file_path = '/mnt/datasets/huichenchen/robosense/rs32/test/32_yueanerlu_1190711151651_990_labels.bin'
json_file_path = '/mnt/datasets/huichenchen/robosense/rs32/test/32_yueanerlu_1190711151651_990.json'
points = read_bin(bin_file_path)
predictions = read_json(json_file_path)

# 筛选标签不为 19 的点云
filtered_points = filter_points(points)

# 获取筛选后点云的真实标签和预测标签
ground_truth_labels = filtered_points[:, 3]
filtered_indices = np.where(points[:, 3] != 19)[0]
filtered_predictions = np.array(predictions)[filtered_indices]

# 计算 IoU 指标（假设类别数为 20）
num_classes = 20
ious = calculate_iou(ground_truth_labels, filtered_predictions, num_classes)

# 打印结果
for class_id, iou in enumerate(ious):
    print(f'Class {class_id} IoU: {iou:.4f}')

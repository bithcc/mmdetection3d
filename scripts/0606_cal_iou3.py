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

# 重映射标签，将多个标签合并为一个类别
def remap_labels(labels, merge_groups):
    remapped_labels = labels.copy()
    for new_label, old_labels in merge_groups.items():
        for label in old_labels:
            remapped_labels[labels == label] = new_label
    return remapped_labels

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
bin_file_path = '/mnt/datasets/huichenchen/robosense/rs32/label_bin/32_32B413_yangyilu_1191014155945_195_labels.bin'
json_file_path = '/mnt/datasets/huichenchen/robosense/rs32/multi-json/preds/32_32B413_yangyilu_1191014155945_195.json'

points = read_bin(bin_file_path)
predictions = read_json(json_file_path)

# 筛选标签不为 19 的点云
filtered_points = filter_points(points)

# 获取筛选后点云的真实标签和预测标签
ground_truth_labels = filtered_points[:, 3]
filtered_indices = np.where(points[:, 3] != 19)[0]
filtered_predictions = np.array(predictions)[filtered_indices]

# 重映射标签
merge_groups = {
    20: [1, 2, 5, 6, 7],  # 合并为类别1
    21: [0, 3, 4],         # 合并为类别0
    # 保留其他类别不变
}

ground_truth_labels = remap_labels(ground_truth_labels, merge_groups)
filtered_predictions = remap_labels(filtered_predictions, merge_groups)

# 计算 IoU 指标（假设类别数为 3，因为合并后有3个有效类别）
num_classes = 22
ious = calculate_iou(ground_truth_labels, filtered_predictions, num_classes)

# 打印结果
for class_id, iou in enumerate(ious):
    print(f'Class {class_id} IoU: {iou:.4f}')

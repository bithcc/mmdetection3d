import os
import numpy as np
import json
from tqdm import tqdm

# 读取 bin 文件
def read_bin(file_path):
    data = np.fromfile(file_path, dtype=np.float32)
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
            iou = np.nan  # 标记为 NaN，后续处理时会忽略
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

# 读取所有 bin 和 json 文件
def read_all_files(bin_folder, json_folder):
    bin_files = [f for f in os.listdir(bin_folder) if f.endswith('_labels.bin')]
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    bin_files_no_ext = {os.path.splitext(f)[0].rsplit('_labels', 1)[0] for f in bin_files}
    json_files_no_ext = {os.path.splitext(f)[0] for f in json_files}
    common_files = bin_files_no_ext.intersection(json_files_no_ext)
    return common_files

# 主函数
def main(bin_folder, json_folder, output_file):
    num_classes = 22
    total_ious = np.zeros(num_classes)
    valid_counts = np.zeros(num_classes)
    num_files = 0

    common_files = read_all_files(bin_folder, json_folder)

    with open(output_file, 'w') as f_out:
        for file_name in tqdm(common_files, desc='Processing files'):
            bin_path = os.path.join(bin_folder, f"{file_name}_labels.bin")
            json_path = os.path.join(json_folder, f"{file_name}.json")
            
            points = read_bin(bin_path)
            predictions = read_json(json_path)
            
            filtered_points = filter_points(points)
            ground_truth_labels = filtered_points[:, 3]
            filtered_indices = np.where(points[:, 3] != 19)[0]
            filtered_predictions = np.array(predictions)[filtered_indices]
            
            ious = calculate_iou(ground_truth_labels, filtered_predictions, num_classes)
            
            f_out.write(f'File: {file_name}\n')
            for class_id, iou in enumerate(ious):
                if not np.isnan(iou):
                    total_ious[class_id] += iou
                    valid_counts[class_id] += 1
                f_out.write(f'Class {class_id} IoU: {iou:.4f}\n')
            
            num_files += 1
        
        avg_ious = np.divide(total_ious, valid_counts, out=np.zeros_like(total_ious), where=valid_counts!=0)
        
        f_out.write('\nAverage IoU:\n')
        for class_id, iou in enumerate(avg_ious):
            f_out.write(f'Class {class_id} IoU: {iou:.4f}\n')


# 示例调用
bin_folder = '/mnt/datasets/huichenchen/robosense/rs128/label_bin'
json_folder = '/mnt/datasets/huichenchen/robosense/rs128/multi-json/preds'
output_file = '/mnt/datasets/huichenchen/robosense/rs128/rs128-multi-iou_results_divide.txt'
main(bin_folder, json_folder, output_file)

import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

# 定义文件路径
bin_file_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/bin/ruby112_lishanlu_1200430192539_2265.bin'
json_file_path_algo1 = '/mnt/datasets/huichenchen/robosense/rs128/paint/multi-json/ruby112_lishanlu_1200430192539_2265.json'
json_file_path_algo2 = '/mnt/datasets/huichenchen/robosense/rs128/paint/cylin-json/ruby112_lishanlu_1200430192539_2265.json'
bin_file_path_with_labels = '/mnt/datasets/huichenchen/robosense/rs128/paint/label_bin/ruby112_lishanlu_1200430192539_2265_labels.bin'  # 包含点云坐标和标签的bin文件

# 读取点云数据
point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)[:, :-1]
point_cloud = point_cloud.astype(np.float64)

# 读取两种算法的标签数据
with open(json_file_path_algo1, 'r') as f:
    labels_algo1 = json.load(f)['pts_semantic_mask']
with open(json_file_path_algo2, 'r') as f:
    labels_algo2 = json.load(f)['pts_semantic_mask']

# 读取标签数据
point_cloud_with_labels = np.fromfile(bin_file_path_with_labels, dtype=np.float32).reshape(-1, 4)
ground_truth_labels = point_cloud_with_labels[:, 3].astype(int)

# 过滤出指定类别的点云及其相应的预测结果
filtered_classes = [0, 1, 2, 3, 5, 18]
mask = np.isin(ground_truth_labels, filtered_classes)
point_cloud_filtered = point_cloud[mask]
ground_truth_labels_filtered = ground_truth_labels[mask]
labels_algo1_filtered = np.array(labels_algo1)[mask]
labels_algo2_filtered = np.array(labels_algo2)[mask]

# 合并类别为car和person
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label in [0, 3, 4]:
            merged_labels.append('car')
        elif label in [1, 2, 5, 6, 7]:
            merged_labels.append('person')
        else:
            merged_labels.append('other')
    return merged_labels

# 合并真实标签和预测结果的类别
merged_ground_truth_labels = merge_classes(ground_truth_labels_filtered)
merged_labels_algo1 = merge_classes(labels_algo1_filtered)
merged_labels_algo2 = merge_classes(labels_algo2_filtered)


merged_ground_truth_labels = np.array(merged_ground_truth_labels)
merged_labels_algo1 = np.array(merged_labels_algo1)
merged_labels_algo2 = np.array(merged_labels_algo2)

# 过滤掉没有真实样本或预测样本的类别
def filter_classes(y_true, y_pred):
    unique_classes = set(y_true).union(set(y_pred))
    filtered_indices = [i for i in unique_classes if np.sum(y_true == i) > 0 or np.sum(y_pred == i) > 0]
    mask = np.isin(y_true, filtered_indices)
    y_true_filtered = y_true[np.where(mask)[0]]
    y_pred_filtered = y_pred[np.where(mask)[0]]
    return y_true_filtered, y_pred_filtered, filtered_indices




# 过滤掉没有真实样本或预测样本的类别
merged_ground_truth_labels_car_person, merged_labels_algo1_car_person, _ = filter_classes(merged_ground_truth_labels, merged_labels_algo1)
merged_ground_truth_labels_car_person, merged_labels_algo2_car_person, _ = filter_classes(merged_ground_truth_labels, merged_labels_algo2)

# 计算性能指标
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

accuracy_algo1_car_person, precision_algo1_car_person, recall_algo1_car_person, f1_algo1_car_person = calculate_metrics(merged_ground_truth_labels_car_person, merged_labels_algo1_car_person)
accuracy_algo2_car_person, precision_algo2_car_person, recall_algo2_car_person, f1_algo2_car_person = calculate_metrics(merged_ground_truth_labels_car_person, merged_labels_algo2_car_person)

# 绘制性能指标柱状图
metrics = ['准确度', '精度', '召回率', 'F1分数']
values_algo1 = [accuracy_algo1_car_person, precision_algo1_car_person, recall_algo1_car_person, f1_algo1_car_person]
values_algo2 = [accuracy_algo2_car_person, precision_algo2_car_person, recall_algo2_car_person, f1_algo2_car_person]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values_algo1, width, label='本文方法')
rects2 = ax.bar(x + width/2, values_algo2, width, label='cylinder3d')

ax.set_xlabel('性能指标',fontsize=12)
ax.set_title('在车辆类与行人类上不同方法的性能对比')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0617_paint/'+'metrics_car_person'+'.png')



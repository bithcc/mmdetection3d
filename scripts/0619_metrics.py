import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

# 定义文件夹路径
bin_folder_path = '/mnt/datasets/huichenchen/robosense/rs32/bin'
json_folder_path_algo1 = '/mnt/datasets/huichenchen/robosense/rs32/multi-json/preds'
json_folder_path_algo2 = '/mnt/datasets/huichenchen/robosense/rs32/cylin-json/preds'
bin_folder_path_with_labels = '/mnt/datasets/huichenchen/robosense/rs32/label_bin'  # 包含点云坐标和标签的bin文件

# 获取文件列表
bin_files = sorted(os.listdir(bin_folder_path))
json_files_algo1 = sorted(os.listdir(json_folder_path_algo1))
json_files_algo2 = sorted(os.listdir(json_folder_path_algo2))
bin_files_with_labels = sorted(os.listdir(bin_folder_path_with_labels))

# 过滤出指定类别的点云及其相应的预测结果
filtered_classes = [0, 1, 2, 3, 5, 18]

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

# 过滤掉没有真实样本或预测样本的类别
def filter_classes(y_true, y_pred):
    unique_classes = set(y_true).union(set(y_pred))
    filtered_indices = [i for i in unique_classes if np.sum(y_true == i) > 0 or np.sum(y_pred == i) > 0]
    mask = np.isin(y_true, filtered_indices)
    y_true_filtered = y_true[np.where(mask)[0]]
    y_pred_filtered = y_pred[np.where(mask)[0]]
    return y_true_filtered, y_pred_filtered, filtered_indices

# 计算性能指标
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

# 初始化指标列表
accuracy_list_algo1 = []
precision_list_algo1 = []
recall_list_algo1 = []
f1_list_algo1 = []

accuracy_list_algo2 = []
precision_list_algo2 = []
recall_list_algo2 = []
f1_list_algo2 = []

# 遍历所有文件进行计算
for bin_file in tqdm(bin_files, total=len(bin_files)):
    file_prefix = os.path.splitext(bin_file)[0]
    bin_file_path = os.path.join(bin_folder_path, bin_file)
    json_file_path_algo1 = os.path.join(json_folder_path_algo1, file_prefix + '.json')
    json_file_path_algo2 = os.path.join(json_folder_path_algo2, file_prefix + '.json')
    bin_file_path_with_labels = os.path.join(bin_folder_path_with_labels, file_prefix + '_labels.bin')

    # 检查文件是否存在，如果任何一个文件不存在，则跳过这次计算
    if not (os.path.exists(bin_file_path) and os.path.exists(json_file_path_algo1) and os.path.exists(json_file_path_algo2) and os.path.exists(bin_file_path_with_labels)):
        print(f"Skipping {file_prefix} due to missing file.")
        continue

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
    mask = np.isin(ground_truth_labels, filtered_classes)
    point_cloud_filtered = point_cloud[mask]
    ground_truth_labels_filtered = ground_truth_labels[mask]
    labels_algo1_filtered = np.array(labels_algo1)[mask]
    labels_algo2_filtered = np.array(labels_algo2)[mask]

    # 合并真实标签和预测结果的类别
    merged_ground_truth_labels = merge_classes(ground_truth_labels_filtered)
    merged_labels_algo1 = merge_classes(labels_algo1_filtered)
    merged_labels_algo2 = merge_classes(labels_algo2_filtered)

    merged_ground_truth_labels = np.array(merged_ground_truth_labels)
    merged_labels_algo1 = np.array(merged_labels_algo1)
    merged_labels_algo2 = np.array(merged_labels_algo2)

    # 过滤掉没有真实样本或预测样本的类别
    merged_ground_truth_labels_car_person, merged_labels_algo1_car_person, _ = filter_classes(merged_ground_truth_labels, merged_labels_algo1)
    merged_ground_truth_labels_car_person, merged_labels_algo2_car_person, _ = filter_classes(merged_ground_truth_labels, merged_labels_algo2)

    # 计算性能指标并加入列表
    accuracy_algo1_car_person, precision_algo1_car_person, recall_algo1_car_person, f1_algo1_car_person = calculate_metrics(merged_ground_truth_labels_car_person, merged_labels_algo1_car_person)
    accuracy_list_algo1.append(accuracy_algo1_car_person)
    precision_list_algo1.append(precision_algo1_car_person)
    recall_list_algo1.append(recall_algo1_car_person)
    f1_list_algo1.append(f1_algo1_car_person)

    accuracy_algo2_car_person, precision_algo2_car_person, recall_algo2_car_person, f1_algo2_car_person = calculate_metrics(merged_ground_truth_labels_car_person, merged_labels_algo2_car_person)
    accuracy_list_algo2.append(accuracy_algo2_car_person)
    precision_list_algo2.append(precision_algo2_car_person)
    recall_list_algo2.append(recall_algo2_car_person)
    f1_list_algo2.append(f1_algo2_car_person)

# 计算平均值
accuracy_avg_algo1 = np.mean(accuracy_list_algo1)
precision_avg_algo1 = np.mean(precision_list_algo1)
recall_avg_algo1 = np.mean(recall_list_algo1)
f1_avg_algo1 = np.mean(f1_list_algo1)

accuracy_avg_algo2 = np.mean(accuracy_list_algo2)
precision_avg_algo2 = np.mean(precision_list_algo2)
recall_avg_algo2 = np.mean(recall_list_algo2)
f1_avg_algo2 = np.mean(f1_list_algo2)

# 绘制性能指标柱状图
metrics = ['准确度', '精度', '召回率', 'F1分数']
values_algo1 = [accuracy_avg_algo1, precision_avg_algo1, recall_avg_algo1, f1_avg_algo1]
values_algo2 = [accuracy_avg_algo2, precision_avg_algo2, recall_avg_algo2, f1_avg_algo2]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values_algo1, width, label='本文方法')
rects2 = ax.bar(x + width/2, values_algo2, width, label='cylinder3d')

ax.set_xlabel('32线所有点云场景性能指标', fontsize=12)
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
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0619_paint/'+'32allmetrics_car_person'+'.png')


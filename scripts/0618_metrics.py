import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

# 定义文件路径
bin_file_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/bin/ruby112_lishanlu_1200430192539_2265.bin'
json_file_path_algo1 = '/mnt/datasets/huichenchen/robosense/rs128/paint/multi-json/ruby112_lishanlu_1200430192539_2265.json'
json_file_path_algo2 = '/mnt/datasets/huichenchen/robosense/rs128/paint/cylin-json/ruby112_lishanlu_1200430192539_2265.json'
bin_file_path_with_labels = '/mnt/datasets/huichenchen/robosense/rs128/paint/label_bin/ruby112_lishanlu_1200430192539_2265_labels.bin'

# 读取点云数据和标签数据的函数
def read_point_cloud_data(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :-1].astype(np.float64)

def read_labels_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['pts_semantic_mask'])

# 读取数据
point_cloud = read_point_cloud_data(bin_file_path)
labels_algo1 = read_labels_from_json(json_file_path_algo1)
labels_algo2 = read_labels_from_json(json_file_path_algo2)
point_cloud_with_labels = np.fromfile(bin_file_path_with_labels, dtype=np.float32).reshape(-1, 4)
ground_truth_labels = point_cloud_with_labels[:, 3].astype(int)

# 过滤出指定类别的点云及其相应的预测结果
filtered_classes = [0, 1, 2, 3, 5, 18]
mask = np.isin(ground_truth_labels, filtered_classes)
point_cloud_filtered = point_cloud[mask]
ground_truth_labels_filtered = ground_truth_labels[mask]
labels_algo1_filtered = labels_algo1[mask]
labels_algo2_filtered = labels_algo2[mask]

# 定义类别映射
class_mapping = {
    '车辆类': {0, 3, 4},
    '行人类': {1, 2, 5, 6, 7}
}

# 合并类别的函数
def merge_classes(labels, class_mapping):
    merged_labels = []
    for label in labels:
        if label in class_mapping['车辆类']:
            merged_labels.append('车辆类')
        elif label in class_mapping['行人类']:
            merged_labels.append('行人类')
        else:
            merged_labels.append('other')  # 其他类别标记为 'other'
    return merged_labels

# 合并类别
merged_ground_truth_labels = merge_classes(ground_truth_labels_filtered, class_mapping)
merged_labels_algo1 = merge_classes(labels_algo1_filtered, class_mapping)
merged_labels_algo2 = merge_classes(labels_algo2_filtered, class_mapping)

# 绘制混淆矩阵的函数
# 绘制混淆矩阵的函数
def plot_confusion_matrix(y_true, y_pred, title):
    classes = ['车辆类', '行人类']  # 只关心 'car' 和 'person' 类别
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 15})
    plt.xlabel('预测结果',fontsize=18)
    plt.ylabel('真值标签',fontsize=18)
    plt.xticks(fontsize=18)  # 设置横坐标文字大小
    plt.yticks(fontsize=18)  # 设置纵坐标文字大小
    plt.title(title,fontsize=18)
    
    # 调整颜色条上数字的大小
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    
    plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0617_paint/'+title+'.png')  # 保存图表的路径
    plt.close()


# 绘制算法1和算法2的混淆矩阵
try:
    plot_confusion_matrix(merged_ground_truth_labels, merged_labels_algo1, '本文算法预测结果的混淆矩阵')
    plot_confusion_matrix(merged_ground_truth_labels, merged_labels_algo2, 'Cylinder3d预测结果的混淆矩阵')
except Exception as e:
    print(f"An error occurred: {e}")





# bin_file_path = '/mnt/datasets/huichenchen/robosense/rs128/paint/bin/ruby112_lishanlu_1200430192539_2265.bin'
# json_file_path_algo1 = '/mnt/datasets/huichenchen/robosense/rs128/paint/multi-json/ruby112_lishanlu_1200430192539_2265.json'
# json_file_path_algo2 = '/mnt/datasets/huichenchen/robosense/rs128/paint/cylin-json/ruby112_lishanlu_1200430192539_2265.json'
# bin_file_path_with_labels = '/mnt/datasets/huichenchen/robosense/rs128/paint/label_bin/ruby112_lishanlu_1200430192539_2265_labels.bin'  # 包含点云坐标和标签的bin文件

# plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0617_paint/'+title+'.png')


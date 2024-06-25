import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

def read_iou_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    scene_names = []
    class_20_ious = []
    class_21_ious = []

    current_scene = ""
    for line in tqdm(data, desc=f"Processing {file_path}"):
        if line.startswith("File:"):
            current_scene = line.split()[1]
            scene_names.append(current_scene)
        elif "Class 20 IoU:" in line:
            class_20_iou = float(line.split()[-1])
            class_20_ious.append(class_20_iou)
        elif "Class 21 IoU:" in line:
            class_21_iou = float(line.split()[-1])
            class_21_ious.append(class_21_iou)

    return scene_names, class_20_ious, class_21_ious

# 文件路径
file_path_1 = '/home/ps/huichenchen/mmdetection3d/results2/0609_paint/rs128-cylin-iou_results.txt'
file_path_2 = '/home/ps/huichenchen/mmdetection3d/results2/0609_paint/rs128-multi-iou_results.txt'

# 读取两个文件的IoU数据
scene_names_1, class_20_ious_1, class_21_ious_1 = read_iou_data(file_path_1)
scene_names_2, class_20_ious_2, class_21_ious_2 = read_iou_data(file_path_2)

# 确保数据长度一致
assert len(scene_names_1) == len(class_20_ious_1) == len(class_21_ious_1)
assert len(scene_names_2) == len(class_20_ious_2) == len(class_21_ious_2)

# 将两种类别分别排序
sorted_class_20_indices_1 = np.argsort(class_20_ious_1)
sorted_class_21_indices_1 = np.argsort(class_21_ious_1)

sorted_class_20_ious_1 = np.array(class_20_ious_1)[sorted_class_20_indices_1]
sorted_class_21_ious_1 = np.array(class_21_ious_1)[sorted_class_21_indices_1]

sorted_class_20_indices_2 = np.argsort(class_20_ious_2)
sorted_class_21_indices_2 = np.argsort(class_21_ious_2)

sorted_class_20_ious_2 = np.array(class_20_ious_2)[sorted_class_20_indices_2]
sorted_class_21_ious_2 = np.array(class_21_ious_2)[sorted_class_21_indices_2]

# 使用插值对齐数据
x_new = np.linspace(0, 1, max(len(sorted_class_20_ious_1), len(sorted_class_20_ious_2)))

sorted_class_20_ious_1_interp = np.interp(x_new, np.linspace(0, 1, len(sorted_class_20_ious_1)), sorted_class_20_ious_1)
sorted_class_21_ious_1_interp = np.interp(x_new, np.linspace(0, 1, len(sorted_class_21_ious_1)), sorted_class_21_ious_1)

sorted_class_20_ious_2_interp = np.interp(x_new, np.linspace(0, 1, len(sorted_class_20_ious_2)), sorted_class_20_ious_2)
sorted_class_21_ious_2_interp = np.interp(x_new, np.linspace(0, 1, len(sorted_class_21_ious_2)), sorted_class_21_ious_2)

# 绘制图表
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# 绘制 Class 20 IoU 排序后的折线图
ax[0].plot(x_new, sorted_class_20_ious_1_interp, label='行人类 IoU (Cylinder3d)', color='blue', marker='o')
ax[0].plot(x_new, sorted_class_20_ious_2_interp, label='行人类 IoU (本文方法)', color='red', marker='x')
ax[0].set_ylabel('IoU', fontsize=18)
ax[0].set_title('排序后的行人类IoU', fontsize=18)
ax[0].legend(fontsize=15)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_xticks(np.linspace(0, 1, 10))
ax[0].set_xticklabels(np.round(np.linspace(0, 1, 10), 2), fontsize=12)
ax[0].set_xlabel('规范化后的场景索引', fontsize=18)

# 绘制 Class 21 IoU 排序后的折线图
ax[1].plot(x_new, sorted_class_21_ious_1_interp, label='车辆类 IoU (Cylinder3d)', color='blue', marker='o')
ax[1].plot(x_new, sorted_class_21_ious_2_interp, label='车辆类 IoU (本文方法)', color='red', marker='x')
ax[1].set_ylabel('IoU', fontsize=18)
ax[1].set_title('排序后的车辆类IoU', fontsize=18)
ax[1].legend(fontsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].set_xticks(np.linspace(0, 1, 10))
ax[1].set_xticklabels(np.round(np.linspace(0, 1, 10), 2), fontsize=12)
ax[1].set_xlabel('规范化后的场景索引', fontsize=18)

fig.tight_layout()

# 显示图表
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0617_paint/rs128-compare.png')


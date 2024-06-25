import os
import shutil

# 设置源文件夹路径
source_folder = '/home/ps/huichenchen/mmdetection3d/results2/robosense/RS_datasets-128/datasets'  # 替换为你的源文件夹路径
# 设置目标文件夹路径
destination_folder = '/home/ps/huichenchen/mmdetection3d/results2/robosense/RS_datasets-128/label'  # 替换为你的目标文件夹路径

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹及其所有子文件夹
for root, dirs, files in os.walk(source_folder):
    # 检查是否存在名为'label'的子文件夹
    if 'label' in dirs:
        label_folder = os.path.join(root, 'label')
        # 遍历Label文件夹中的所有文件
        for file in os.listdir(label_folder):
            # 构建文件的完整路径
            file_path = os.path.join(label_folder, file)
            # 构建目标文件的完整路径
            destination_file_path = os.path.join(destination_folder, file)
            # 移动文件
            shutil.copy(file_path, destination_file_path)
            print(f'copy: {file_path} to {destination_file_path}')

print('All files have been copyed successfully.')
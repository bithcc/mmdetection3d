import os

# 设置文件夹路径
folder_label_bin = '/mnt/datasets/huichenchen/robosense/rs128/label_bin'
folder_multi_json = '/mnt/datasets/huichenchen/robosense/rs128/multi-json/preds'
folder_cylin_json = '/mnt/datasets/huichenchen/robosense/rs128/cylin-json/preds'

# 获取每个文件夹中的文件列表
files_label_bin = {f[:-11] for f in os.listdir(folder_label_bin) if f.endswith('_labels.bin')}  # 假设文件名以_label.bin结尾
files_multi_json = {f[:-5] for f in os.listdir(folder_multi_json) if f.endswith('.json')}  # 假设文件名以.json结尾
files_cylin_json = {f[:-5] for f in os.listdir(folder_cylin_json) if f.endswith('.json')}  # 同上

# 找出三个文件夹中共有的文件名前缀
common_files = files_label_bin & files_multi_json & files_cylin_json

# 打印结果
print("共有的文件名前缀:")
for file_name in common_files:
    print(file_name)
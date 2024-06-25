import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 定义处理单个点云和标签文件的函数
def process_file(bin_file_path, json_file_path, output_dir):
    try:
        # 读取点云数据
        point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
        point_cloud_coordinates = point_cloud[:, :3]  # 只取前三个坐标

        # 读取JSON标注文件
        with open(json_file_path, 'r') as f:
            annotations = json.load(f)['pts_semantic_mask']

        # 确保点云坐标和标签数量一致
        if point_cloud_coordinates.shape[0] != len(annotations):
            raise ValueError("点云坐标和标签数量不匹配")

        # 合并点云坐标和标签
        point_cloud_with_labels = np.hstack([point_cloud_coordinates, annotations[:, np.newaxis]])

        # 构建输出文件的完整路径
        output_file_path = os.path.join(output_dir, os.path.basename(bin_file_path).replace('.bin', '_with_labels.bin'))

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 将合并后的数据写入新的bin文件
        with open(output_file_path, 'wb') as f:
            point_cloud_with_labels.tofile(f)
        print(f"Processed {bin_file_path} and {json_file_path} -> {output_file_path}")
    except Exception as e:
        print(f"Error processing {bin_file_path} and {json_file_path}: {e}")

# 获取bin和json文件列表
bin_dir = 'path/to/bin/files'
json_dir = 'path/to/json/files'
output_dir = 'path/to/output/files'  # 指定输出文件夹

bin_files = [os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith('.bin')]
json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]

# 确保bin和json文件数量一致
if len(bin_files) != len(json_files):
    print("不同数量的bin和json文件")
else:
    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor() as executor:
        # 使用tqdm来显示进度条
        futures = [executor.submit(process_file, bin_files[i], json_files[i], output_dir) for i in tqdm(range(len(bin_files)))]

    # 等待所有任务完成
    for future in tqdm(futures, desc="Waiting for all tasks to complete"):
        future.result()
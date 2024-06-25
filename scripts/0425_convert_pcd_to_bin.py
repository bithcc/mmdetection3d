import numpy as np
import open3d as o3d
import glob
import os
from tqdm import tqdm



def pcd_to_bin(pcd_file_path, bin_file_path):
    # 读取PCD文件
    pcd = o3d.t.io.read_point_cloud(pcd_file_path)
    
    # 获取点云数据
    points = pcd.point["positions"]
    intensity = pcd.point["intensity"]
    points = points[:,:].numpy()
    intensity = intensity[:,:].numpy()
    
    normalized_intensity = intensity/255.0
    points = np.hstack((points,normalized_intensity))
    # #robosense  x y z intensity
    # points = points[:,:4]
    # last_column = points[:, 3]
    # normalized_last_column = (last_column - np.min(last_column)) / (np.max(last_column) - np.min(last_column))
    # points[:, 3] = normalized_last_column
    
    # neuvition  x y z rgba time_sec time_usec intensity
    # points = np.hstack((points[:, :3], points[:, -1].reshape(-1, 1)))
    # last_column = points[:, 3]
    # normalized_last_column = (last_column - np.min(last_column)) / (np.max(last_column) - np.min(last_column))
    # points[:, 3] = normalized_last_column
    
    #seyond  x y z timestamp intensity flags scan_id scan_idx
    # points = np.hstack((points[:, :3], points[:, 4].reshape(-1, 1)))
    # last_column = points[:, 3]
    # normalized_last_column = (last_column - np.min(last_column)) / (np.max(last_column) - np.min(last_column))
    # points[:, 3] = normalized_last_column
    
    
    # 保存点云数据到BIN文件
    points.astype(np.float32).tofile(bin_file_path)

# def batch_convert_pcd_to_bin(source_folder, target_folder):
#     # 确保目标文件夹存在
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
    
#     # 获取源文件夹中所有的PCD文件
#     pcd_files = glob.glob(os.path.join(source_folder, '*.pcd'))
    
#     for pcd_file in pcd_files:
#         # 构建目标BIN文件的路径
#         file_name = os.path.basename(pcd_file).replace('.pcd', '.bin')
#         bin_file_path = os.path.join(target_folder, file_name)
        
#         # 转换PCD到BIN并添加默认强度信息
#         pcd_to_bin(pcd_file, bin_file_path)
#         print(f"Converted {pcd_file} to {bin_file_path} with default intensities.")

def batch_convert_pcd_to_bin(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取源文件夹中所有的PCD文件
    pcd_files = glob.glob(os.path.join(source_folder, '*.pcd'))
    
    # 使用tqdm包装pcd_files迭代器来显示进度条
    for pcd_file in tqdm(pcd_files, desc='Converting PCD files'):
        # 构建目标BIN文件的路径
        file_name = os.path.basename(pcd_file).replace('.pcd', '.bin')
        bin_file_path = os.path.join(target_folder, file_name)
        
        # 转换PCD到BIN并添加默认强度信息
        pcd_to_bin(pcd_file, bin_file_path)
        # print(f"Converted {pcd_file} to {bin_file_path} with default intensities.")

# 源文件夹和目标文件夹的路径
source_folder = '/mnt/datasets/huichenchen/robosense/rs128/pcd'
target_folder = '/mnt/datasets/huichenchen/robosense/rs128/bin'

# 批量转换
batch_convert_pcd_to_bin(source_folder, target_folder)

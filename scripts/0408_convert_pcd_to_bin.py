import numpy as np
import open3d as o3d
import glob
import os

def create_rotation_matrix_x(angle):
    """根据给定的角度(以度为单位),创建绕X轴的旋转矩阵。"""
    rad = np.radians(angle)  # 将角度转换为弧度
    cos = np.cos(rad)
    sin = np.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, cos, -sin],
        [0, sin, cos]
    ])

def rotate_point_cloud(points, rotation_matrix):
    """使用给定的旋转矩阵旋转点云。"""
    return np.dot(points, rotation_matrix.T)
# transform_matrix = np.array([
#     [0.999862951, -0.0013060762, 0.052319661, 0.0299999999],
#     [0, 0.99968857, 0.024955617, 0.039999999],
#     [-0.052335959, -0.024921415, 0.99831849, 0.05000001]
# ],dtype=np.float32)

# def transform_points_homogeneous(points,transform_matrix):
#     points_homogeneous = np.hstack((points,np.ones((points.shape[0],1),dtype=np.float32)))
#     points_transformed_homogeneous = np.dot(points_homogeneous,transform_matrix.T)
#     points_transformed = points_transformed_homogeneous[:,:3]
#     return points_transformed

def pcd_to_bin(pcd_file_path, bin_file_path):
    # 读取PCD文件
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    # 获取点云数据
    points = np.asarray(pcd.points)
    # points = transform_points_homogeneous(points, transform_matrix)
    # points = points[:,[2,1,0]]

    rotation_matrix = create_rotation_matrix_x(150)
    points = rotate_point_cloud(points, rotation_matrix)

    
    # 创建一个强度值为1的数组，长度与点云中的点的数量相同
    intensities = np.ones((points.shape[0], 1), dtype=np.float32)
    
    # 将强度信息附加到点云数据中
    points_with_intensities = np.hstack((points, intensities))
    
    # 保存点云数据到BIN文件
    points_with_intensities.astype(np.float32).tofile(bin_file_path)

def batch_convert_pcd_to_bin(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取源文件夹中所有的PCD文件
    pcd_files = glob.glob(os.path.join(source_folder, '*.pcd'))
    
    for pcd_file in pcd_files:
        # 构建目标BIN文件的路径
        file_name = os.path.basename(pcd_file).replace('.pcd', '.bin')
        bin_file_path = os.path.join(target_folder, file_name)
        
        # 转换PCD到BIN并添加默认强度信息
        pcd_to_bin(pcd_file, bin_file_path)
        print(f"Converted {pcd_file} to {bin_file_path} with default intensities.")

# 源文件夹和目标文件夹的路径
source_folder = '/home/ps/huichenchen/mmdetection3d/results2/test2'
target_folder = '/home/ps/huichenchen/mmdetection3d/results2/test2'

# 批量转换
batch_convert_pcd_to_bin(source_folder, target_folder)

import numpy as np

def load_bin_file(bin_file_path):
    # 使用 np.fromfile 来读取二进制文件
    # dtype='float32' 表示文件中的数据类型是 32 位浮点数
    # reshape(-1, 3) 将数据重塑为 (N, 3) 形状的数组，其中 N 是点的数量
    # 每行包含一个点的 x、y、z 坐标
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return points

# 指定 .bin 文件的路径
bin_file_path = '/mnt/datasets/huichenchen/robosense/rs128/test/label_bin/ruby119_nanshandadao_1200421163451_5600.bin'

# 读取点云数据
point_cloud = load_bin_file(bin_file_path)
print('aa')

# 现在 point_cloud 数组中包含了所有点的坐标
# 例如，point_cloud[0] 将给出第一个点的坐标
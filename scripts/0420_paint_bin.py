import numpy as np
import open3d as o3d

def load_bin_point_cloud(bin_path):
    """加载BIN格式的点云文件"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # 假设每个点带有4个属性，X, Y, Z 和强度
    return points[:, :3]  # 只返回X, Y, Z

def add_color_to_points(points, color=[1, 0, 0]):  # 默认为红色
    """为点云添加颜色"""
    colors = np.tile(color, (points.shape[0], 1))  # 创建一个与点云数量相同的颜色数组
    return colors

def save_to_ply(points, colors, output_path):
    """将带颜色的点云数据保存为PLY格式"""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, point_cloud)

def main():
    bin_path = "/home/ps/huichenchen/mmdetection3d/scripts/002500.bin"  # 替换为您的点云文件路径
    output_path = "/home/ps/huichenchen/mmdetection3d/results2/0420_002500_paint.ply"  # 输出PLY文件路径

    # 加载点云数据
    points = load_bin_point_cloud(bin_path)
    # 为点云添加颜色
    colors = add_color_to_points(points, [0, 0, 1])  
    # 保存处理后的带颜色的点云为PLY文件
    save_to_ply(points, colors, output_path)
    print("Colored point cloud has been saved to:", output_path)

if __name__ == "__main__":
    main()

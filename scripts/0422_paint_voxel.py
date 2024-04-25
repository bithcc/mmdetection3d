import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_bin_point_cloud(file_path):
    """Load point cloud from a .bin file."""
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))[:, :3]
    return points

def voxelize_point_cloud(points, voxel_size=0.1):
    """Convert points to a voxel grid."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def plot_voxel_grid(voxel_grid, voxel_size, output_file):
    """Plot the voxel grid and save as an image."""
    voxels = np.asarray(voxel_grid.get_voxels())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

    # Draw each voxel
    for voxel in tqdm(voxels, desc='Plotting voxels'):
        x, y, z = voxel.grid_index
        ax.bar3d(x * voxel_size, y * voxel_size, z * voxel_size, voxel_size, voxel_size, voxel_size, color='b', edgecolor='k')

    plt.axis('off')  # Turn off the axis
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    file_path = '/home/ps/huichenchen/mmdetection3d/scripts/000000.bin'
    output_image = '/home/ps/huichenchen/mmdetection3d/results2/0422_paint4.png'
    voxel_size = 2

    points = load_bin_point_cloud(file_path)
    voxel_grid = voxelize_point_cloud(points, voxel_size)
    plot_voxel_grid(voxel_grid, voxel_size, output_image)

    print(f"Saved voxel grid image to {output_image}")

if __name__ == "__main__":
    main()

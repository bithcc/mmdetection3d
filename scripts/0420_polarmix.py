import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    """Load a point cloud from a binary file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Assuming points are [x, y, z, intensity]

def polar_sector_mix(points_a, points_b):
    """Mix two point clouds by selecting a sector from one and the rest from the other."""
    start_angle = (np.random.random() - 1) * np.pi  # -pi~0
    end_angle = start_angle + np.pi
    angle_range = (start_angle,end_angle)
    # Calculate angles for points in both clouds
    angles_a = np.arctan2(points_a[:, 1], points_a[:, 0])
    angles_b = np.arctan2(points_b[:, 1], points_b[:, 0])

    # Define masks for the sector and outside the sector
    sector_mask_a = (angles_a >= angle_range[0]) & (angles_a <= angle_range[1])
    outside_sector_mask_b = (angles_b < angle_range[0]) | (angles_b > angle_range[1])

    # Select points based on the masks
    sector_points = points_a[sector_mask_a]
    outside_sector_points = points_b[outside_sector_mask_b]

    # Colors: Red for sector, Green for outside sector
    sector_colors = np.array([[1, 0, 0]] * len(sector_points))
    outside_sector_colors = np.array([[0, 0, 1]] * len(outside_sector_points))

    # Combine points and colors
    combined_points = np.vstack((sector_points, outside_sector_points))
    combined_colors = np.vstack((sector_colors, outside_sector_colors))

    return combined_points, combined_colors

def save_to_ply(points, colors, filename):
    """Save colored point cloud to a PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def main():
    # File paths
    file_path_a = '/home/ps/huichenchen/mmdetection3d/scripts/000000.bin'
    file_path_b = '/home/ps/huichenchen/mmdetection3d/scripts/002500.bin'
    output_file = '/home/ps/huichenchen/mmdetection3d/results2/polarmix3.ply'
    
    # Load point clouds
    points_a = load_point_cloud(file_path_a)
    points_b = load_point_cloud(file_path_b)
    
    # Mix the point clouds
    combined_points, colors = polar_sector_mix(points_a, points_b)
    
    # Save to PLY
    save_to_ply(combined_points, colors, output_file)
    print("Saved combined point cloud to", output_file)

if __name__ == "__main__":
    main()

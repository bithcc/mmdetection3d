import numpy as np
import open3d as o3d

class LaserMix:
    def __init__(self, num_areas, pitch_angles, prob=1.0):
        self.num_areas = num_areas
        self.pitch_angles = np.radians(pitch_angles)  # Convert degrees to radians
        self.prob = prob

    def laser_mix_transform(self, points_a, points_b):
        if np.random.rand() > self.prob:
            return points_a, points_b  # No transformation applied

        # Convert points to cylindrical coordinates
        rho_a = np.sqrt(points_a[:, 0]**2 + points_a[:, 1]**2)
        pitch_a = np.arctan2(points_a[:, 2], rho_a)
        pitch_a = np.clip(pitch_a, self.pitch_angles[0] + 1e-5, self.pitch_angles[1] - 1e-5)

        rho_b = np.sqrt(points_b[:, 0]**2 + points_b[:, 1]**2)
        pitch_b = np.arctan2(points_b[:, 2], rho_b)
        pitch_b = np.clip(pitch_b, self.pitch_angles[0] + 1e-5, self.pitch_angles[1] - 1e-5)

        # Determine the pitch angle ranges for slicing
        angle_ranges = np.linspace(self.pitch_angles[0], self.pitch_angles[1], self.num_areas + 1)

        # Create masks for slicing
        masks_a = [(pitch_a >= angle_ranges[i]) & (pitch_a < angle_ranges[i + 1]) for i in range(self.num_areas)]
        masks_b = [(pitch_b >= angle_ranges[i]) & (pitch_b < angle_ranges[i + 1]) for i in range(self.num_areas)]

        # Interleave the slices
        combined_points = []
        combined_colors = []
        for i in range(self.num_areas):
            if i % 2 == 0:
                combined_points.append(points_a[masks_a[i]])
                combined_colors.append(np.array([1, 0, 0] * sum(masks_a[i])).reshape(-1, 3))  # Red color
            else:
                combined_points.append(points_b[masks_b[i]])
                combined_colors.append(np.array([0, 0, 1] * sum(masks_b[i])).reshape(-1, 3))  # Green color

        combined_points = np.vstack(combined_points)
        combined_colors = np.vstack(combined_colors)

        return combined_points, combined_colors

    def save_to_ply(self, points, colors, filename):
        """Save colored point cloud to a PLY file."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

def load_point_cloud(file_path):
    """Load a point cloud from a binary file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Assuming points are [x, y, z, intensity]

def main():
    # File paths
    file_path_a = '/home/ps/huichenchen/mmdetection3d/scripts/000000.bin'
    file_path_b = '/home/ps/huichenchen/mmdetection3d/scripts/002500.bin'
    output_file = '/home/ps/huichenchen/mmdetection3d/results2/lasermix_6part.ply'
    
    # Load point clouds
    points_a = load_point_cloud(file_path_a)
    points_b = load_point_cloud(file_path_b)
    
    # Create a LaserMix object
    mixer = LaserMix(num_areas=6, pitch_angles=[-25, 3])  # Example angles in degrees
    
    # Mix point clouds and get colors
    combined_points, colors = mixer.laser_mix_transform(points_a, points_b)
    
    # Save to PLY
    mixer.save_to_ply(combined_points, colors, output_file)
    print("Saved combined point cloud to", output_file)

if __name__ == "__main__":
    main()

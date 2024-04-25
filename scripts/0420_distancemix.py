import numpy as np
import open3d as o3d

class DistanceMix:
    def __init__(self, distance_bins, swap_ratio=0.5, prob=1.0):
        self.distance_bins = np.array(distance_bins)  # Distances to define segments
        self.swap_ratio = swap_ratio  # Probability to swap each segment
        self.prob = prob  # Overall probability to apply the transformation

    def distance_mix_transform(self, points_a, points_b):
        if np.random.rand() > self.prob:
            return np.vstack((points_a, points_b)), None  # No transformation applied

        # Compute radial distances for each point
        distances_a = np.sqrt(np.sum(points_a[:, :2] ** 2, axis=1))
        distances_b = np.sqrt(np.sum(points_b[:, :2] ** 2, axis=1))

        # Binning points by distance
        bin_indices_a = np.digitize(distances_a, self.distance_bins)
        bin_indices_b = np.digitize(distances_b, self.distance_bins)

        combined_points = []
        colors = []

        # Interleave segments from both point clouds based on distance bins
        for bin_index in range(1, len(self.distance_bins) + 1):
            mask_a = bin_indices_a == bin_index
            mask_b = bin_indices_b == bin_index

            if np.random.rand() < self.swap_ratio:
                # Swap points from points_b to points_a
                combined_points.append(points_b[mask_b])
                colors.extend([[0, 0, 1]] * len(points_b[mask_b]))  # Green for points from B
            else:
                # Keep points from points_a
                combined_points.append(points_a[mask_a])
                colors.extend([[1, 0, 0]] * len(points_a[mask_a]))  # Red for points from A

        combined_points = np.vstack(combined_points)
        colors = np.array(colors)

        return combined_points, colors

    def save_to_ply(self, points, colors, filename):
        """Save points and colors to a PLY file."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

def load_bin_point_cloud(file_path):
    """Load a point cloud from a binary file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Assuming points are [x, y, z]

def main():
    # File paths
    file_path_a = '/home/ps/huichenchen/mmdetection3d/scripts/000000.bin'
    file_path_b = '/home/ps/huichenchen/mmdetection3d/scripts/002500.bin'
    output_file = '/home/ps/huichenchen/mmdetection3d/results2/distancemix5.ply'

    # Load point clouds
    points_a = load_bin_point_cloud(file_path_a)
    points_b = load_bin_point_cloud(file_path_b)

    # Create a DistanceMix object with specified distance bins
    distance_bins = [10, 20, 30, 40,50]  # Define bins edges
    mixer = DistanceMix(distance_bins, swap_ratio=0.5)

    # Mix point clouds and get colors
    mixed_points, colors = mixer.distance_mix_transform(points_a, points_b)

    # Save to PLY
    mixer.save_to_ply(mixed_points, colors, output_file)
    print(f"Saved mixed point cloud to {output_file}")

if __name__ == "__main__":
    main()

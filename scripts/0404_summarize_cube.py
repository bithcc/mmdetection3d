import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_bin_files(directory):
    """Get all .bin files in the directory and its subdirectories."""
    bin_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    return bin_files

def process_point_cloud(file_path, length_voxel_size, width_voxel_size):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    voxels = set()
    for point in points:
        x=abs(point[0])
        y=abs(point[1])
        # z = point[2]
        voxel = (
            np.floor(x / length_voxel_size),
            np.floor(y / width_voxel_size),
            # np.floor(z / height_voxel_size)
        )
        voxels.add(voxel)
    return voxels

def calculate_voxel_proportions(bin_files, length_voxel_size, width_voxel_size, distance_ranges):
    voxel_counts = {range_desc: set() for range_desc in distance_ranges}
    theoretical_voxel_counts = {}

    for file_path in tqdm(bin_files, desc="Processing files"):
        file_voxels = process_point_cloud(file_path, length_voxel_size, width_voxel_size)
        for voxel in file_voxels:
            x_index = voxel[0]
            y_index = voxel[1]
            distance = max(x_index*length_voxel_size, y_index*width_voxel_size) 
            for range_desc, (min_dist, max_dist) in distance_ranges.items():
                if min_dist <= distance < max_dist:
                    voxel_counts[range_desc].add(voxel)
                    break

    for range_desc, (min_dist, max_dist) in distance_ranges.items():
        outer = max_dist**2/(length_voxel_size*width_voxel_size)
        inner = min_dist**2/(length_voxel_size*width_voxel_size)
        # num_z_bins = (np.ceil(max_dist / height_voxel_size) - np.floor(min_dist / height_voxel_size))
        theoretical_voxel_counts[range_desc] = outer - inner

    proportions = {
        range_desc: len(voxels) / theoretical_voxel_counts[range_desc]
        for range_desc, voxels in voxel_counts.items()
    }

    return proportions

# Example usage
directory_path = '/home/ps/huichenchen/mmdetection3d/scripts'
# directory_path = '/mnt/datasets/huichenchen/SemanticKitti/dataset/sequences'  # Replace with your directory path
bin_files = get_bin_files(directory_path)
length_voxel_size = 0.1
width_voxel_size = 0.1
distance_ranges = {
    '0-10m': (0, 10),
    '10-20m': (10, 20),
    '20-30m': (20, 30),
    '30-40m': (30, 40),
    '40-50m': (40, 50),
    '50m+': (50, np.inf)
}

proportions = calculate_voxel_proportions(bin_files, length_voxel_size, width_voxel_size, distance_ranges)

# Plotting
bars = plt.bar(proportions.keys(), proportions.values())
plt.ylabel('Proportion of Actual to Theoretical Voxels')
plt.xlabel('Distance Range')
plt.title('Voxel Distribution by Distance')
plt.xticks(rotation=45)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/test_cube.png')


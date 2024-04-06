import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def read_bin_point_cloud(filepath):
    point_cloud = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # Only x, y, z coordinates

def voxelization(points, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(int)
    unique_voxels = set(map(tuple, voxel_indices))
    return unique_voxels

def cylindrical_voxelization(points, voxel_size_rho, voxel_size_phi, voxel_size_z):
    rho = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    phi = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360
    z = points[:, 2]
    voxel_indices = set((int(rho[i] / voxel_size_rho), int(phi[i] / voxel_size_phi), int(z[i] / voxel_size_z))
                        for i in range(len(points)))
    return voxel_indices

def get_bin_files(directory):
    bin_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    return bin_files

# Parameters
# directory_path = '/home/ps/huichenchen/mmdetection3d/scripts'
directory_path = '/mnt/datasets/huichenchen/SemanticKitti/dataset/sequences'
voxel_size_rho = 0.1
voxel_size_phi = 360 / 360  # 1 degree
voxel_size_z = 0.1
standard_voxel_size = 0.1

# Accumulate voxels
cylindrical_voxels_accum = set()
standard_voxels_accum = set()

# Process point clouds
bin_files = get_bin_files(directory_path)
for filepath in tqdm(bin_files, desc="Processing"):
    points = read_bin_point_cloud(filepath)
    cylindrical_voxels_accum |= cylindrical_voxelization(points, voxel_size_rho, voxel_size_phi, voxel_size_z)
    standard_voxels_accum |= voxelization(points, standard_voxel_size)

# Calculate ratios
max_distance = 50  # Maximum distance to consider
distance_step = 1  # Step size for distances
distances = np.arange(0, max_distance + distance_step, distance_step)
cylindrical_ratios = []
standard_ratios = []

total_cylindrical_voxels = len(cylindrical_voxels_accum)
total_standard_voxels = len(standard_voxels_accum)

for distance in distances:
    cylindrical_voxels_at_distance = len({v for v in cylindrical_voxels_accum if (v[0] * voxel_size_rho) < distance})
    standard_voxels_at_distance = len({v for v in standard_voxels_accum if np.linalg.norm(np.array(v)[0:-1] * standard_voxel_size) < distance})
    cylindrical_ratios.append(cylindrical_voxels_at_distance / total_cylindrical_voxels if total_cylindrical_voxels else 0)
    standard_ratios.append(standard_voxels_at_distance / total_standard_voxels if total_standard_voxels else 0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distances, cylindrical_ratios, marker='o', label='Cylindrical Voxelization')
plt.plot(distances, standard_ratios, marker='x', label='Standard Voxelization')
plt.xlabel('Distance (m)')
plt.ylabel('Proportion of Accumulated Valid Voxels')
plt.title('Proportion of Accumulated Valid Voxels by Distance')
plt.legend()
plt.grid(True)
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0406_compare_voxel_cylinder.png')

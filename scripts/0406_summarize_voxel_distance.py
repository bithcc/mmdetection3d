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
# directory_path = '/mnt/datasets/huichenchen/SemanticKitti/dataset/sequences/08'
# directory_path = '/mnt/datasets/huichenchen/SemanticKitti/dataset/sequences'
directory_path = '/home/ps/huichenchen/mmdetection3d/results2/test'
voxel_size_rho = 0.1
voxel_size_phi = 360 / 360  # 1 degree
voxel_size_z = 0.1
standard_voxel_size = 0.1

# Accumulate voxels
cylindrical_voxels_accum = set()
standard_voxels_accum = set()

# Process point clouds
bin_files = get_bin_files(directory_path)
# for filepath in tqdm(bin_files, desc="Processing"):
#     points = read_bin_point_cloud(filepath)
#     cylindrical_voxels_accum |= cylindrical_voxelization(points, voxel_size_rho, voxel_size_phi, voxel_size_z)
#     standard_voxels_accum |= voxelization(points, standard_voxel_size)

# Calculate ratios
max_distance = 50  # Maximum distance to consider
distance_step = 1  # Step size for distances
distances = np.arange(0, max_distance + distance_step, distance_step)
cylindrical_ratios = []
standard_ratios = []

total_cylindrical_voxels = len(cylindrical_voxels_accum)
total_standard_voxels = len(standard_voxels_accum)

# for distance in distances:
#     cylindrical_voxels_at_distance = len({v for v in cylindrical_voxels_accum if (v[0] * voxel_size_rho) < distance})
#     standard_voxels_at_distance = len({v for v in standard_voxels_accum if np.linalg.norm(np.array(v)[0:-1] * standard_voxel_size) < distance})
#     cylindrical_ratios.append(cylindrical_voxels_at_distance / total_cylindrical_voxels if total_cylindrical_voxels else 0)
#     standard_ratios.append(standard_voxels_at_distance / total_standard_voxels if total_standard_voxels else 0)

cylindrical_ratios = [0.0, 0.0, 0.000845874531111481, 0.003385498334467976, 
                      0.007328555212583036, 0.012826132458193825, 0.019676994655867436,
                      0.02723928869674173, 0.0354223622051744, 0.04427150565095049, 
                      0.05356848183344966, 0.06333672178435878, 0.07355700920025174, 
                      0.08412904783573703, 0.09502040571402855, 0.10619657921224586, 
                      0.11762077875319765, 0.12932936529764197, 0.14131776693695694, 
                      0.15354755106429033, 0.1660285044215357, 0.1787687707209257, 
                      0.1917486336065287, 0.20497559386592734, 0.21841664803375782, 
                      0.23201857623621908, 0.2457263726977047, 0.25955396745229675, 
                      0.27339334915880453, 0.2874655410246647, 0.3015183737149544, 
                      0.31554184617956316, 0.3297649110299439, 0.3438309593935935, 
                      0.35807152608166715, 0.37244532104442396, 0.38692769883694933, 
                      0.4014124340198578, 0.41598946460806957, 0.4307512288790323, 
                      0.44545498705935554, 0.460128099164698, 0.47473070386676314, 
                      0.48945775020662885, 0.5042359447742013, 0.5188971270554836, 
                      0.5335212340152858, 0.5482857842930651, 0.5630275464638073, 
                      0.5776160782596458, 0.5923149787870584]

standard_ratios = [0.0, 0.0, 6.530932871484166e-05, 0.00034463205339529235, 
                   0.0009319439354248966, 0.001959556595888957, 0.003511027558290285, 
                   0.005526598461284868, 0.008022743143521612, 0.01107920996027819, 
                   0.01462977798702519, 0.018710992448828626, 0.023403334233317155, 
                   0.028622545841630348, 0.03441186296097325, 0.04079801488290861, 
                   0.04767377514974833, 0.055154460632034774, 0.06321335831670187, 
                   0.07177483830813416, 0.08092756957779472, 0.09065615266781608, 
                   0.1009597736533638, 0.11187276388395424, 0.12324389612534989, 
                   0.13523642742084116, 0.14775477043787236, 0.16074190160254312, 
                   0.17406220475894038, 0.1880290735253139, 0.20222262035807678, 
                   0.21662767369831523, 0.23148946909864368, 0.2463431089721311, 
                   0.2616317591126292, 0.27726818305638345, 0.29334695886915496, 
                   0.30952727994427026, 0.3261924397647601, 0.3435277201799347, 
                   0.3607005086411632, 0.37783562866105425, 0.39493379649346205, 
                   0.4124011606307683, 0.4303236552569924, 0.4478686352665101, 
                   0.4653234324043727, 0.48327630270541777, 0.5012927893715221, 
                   0.5189663857833767, 0.5372006298996851]
# Plotting
plt.figure(figsize=(10, 6))
# plt.plot(distances, cylindrical_ratios, marker='o', label='Cylindrical Voxelization')
# plt.plot(distances, standard_ratios, marker='x', label='Standard Voxelization')
# plt.xlabel('Distance (m)')
# plt.ylabel('Proportion of Accumulated Valid Voxels')
# plt.title('Proportion of Accumulated Valid Voxels by Distance')
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/test/0408_compare08.png')

print(distances)
print('cylindrical_ratios:',cylindrical_ratios)
print('standard_ratios:',standard_ratios)
plt.rcParams['font.sans-serif'] = ['SimHei']#指定字体为SimHei
plt.plot(distances, cylindrical_ratios, marker='o', label='柱面体素分割')
plt.plot(distances, standard_ratios, marker='x', label='标准体素分割')
plt.xlabel('距离（米）',fontsize=15)
plt.ylabel('累计有效体素数占总体素数比例',fontsize=15)
# plt.title('Proportion of Accumulated Valid Voxels by Distance')
plt.legend(fontsize=15)
plt.tick_params(axis='both',which='major',labelsize=12)
plt.grid(True)
plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0523_compare_all_test.png')


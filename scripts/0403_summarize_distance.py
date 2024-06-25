import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']#指定字体为SimHei
def get_bin_files(directory):
    """Recursively find all .bin files within given directory and its subdirectories."""
    bin_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    return bin_files

def read_point_cloud(bin_file):
    """Read point cloud from a binary file."""
    return np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

def calculate_distance_distribution(bin_files):
    """Calculate distance distribution from a list of bin files."""
    # Initialize the distance distribution dictionary
    distance_distribution = {
        '小于 10m': 0,
        '10m - 20m': 0,
        '20m - 30m': 0,
        '30m - 40m': 0,
        '40m - 50m': 0,
        '大于 50m': 0,
    }
    
    # Process each file
    for bin_file in tqdm(bin_files, desc='Processing bin files', unit='file'):
        point_cloud = read_point_cloud(bin_file)
        distances = np.linalg.norm(point_cloud[:, :2], axis=1)  # Ignore intensity
        
        # Increment counts in distance distribution
        distance_distribution['小于 10m'] += np.sum(distances < 10)
        distance_distribution['10m - 20m'] += np.sum((distances >= 10) & (distances < 20))
        distance_distribution['20m - 30m'] += np.sum((distances >= 20) & (distances < 30))
        distance_distribution['30m - 40m'] += np.sum((distances >= 30) & (distances < 40))
        distance_distribution['40m - 50m'] += np.sum((distances >= 40) & (distances < 50))
        distance_distribution['大于 50m'] += np.sum(distances >= 50)
    
    print(distance_distribution)    
    return distance_distribution

def plot_pie_chart(distribution):
    """Plot a pie chart with the given distribution."""
    labels = distribution.keys()
    sizes = distribution.values()
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'violet','cyan']
    
    plt.figure(figsize=(10, 10))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,pctdistance=0.8,textprops={'fontsize':18})
    plt.legend(labels, loc="lower right",fontsize=18)
    plt.axis('equal')
    plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0522_summarize_all_sem.png')

# Example usage:
# Provide the path to the directory containing the .bin files
directory_path = '/mnt/datasets/huichenchen/SemanticKitti/dataset/sequences'
# directory_path = '/home/ps/huichenchen/mmdetection3d/scripts'

# bin_files = get_bin_files(directory_path)
# distribution = calculate_distance_distribution(bin_files)
distribution = {
        '小于 10m': 0,
        '10m - 20m': 0,
        '20m - 30m': 0,
        '30m - 40m': 0,
        '40m - 50m': 0,
        '大于 50m': 0,
    }
distribution['小于 10m'] = 3239483528
distribution['10m - 20m'] = 1372513773
distribution['20m - 30m'] = 367781986
distribution['30m - 40m'] = 154416096
distribution['40m - 50m'] = 79644899
distribution['大于 50m'] = 84720742
plot_pie_chart(distribution)

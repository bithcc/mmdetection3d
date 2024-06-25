import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_label_file(label_file, label_map):
    """读取.label格式的标签数据,并应用标签映射"""
    try:
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF  # 获取语义标签
        mapped_labels = np.array([label_map.get(label, label) for label in semantic_labels])
        return mapped_labels
    except Exception as e:
        print(f"读取文件出错: {label_file}, 错误: {str(e)}")
        return np.array([])  # 返回一个空数组以避免进一步错误

def count_labels(folder_paths, label_map, category_map):
    """
    读取指定文件夹路径列表中的所有标签文件并计数，返回类别名及其计数。
    
    Args:
    - folder_paths (list of str): 包含标签文件的文件夹路径列表。
    - label_map (dict): 从原始标签到新标签的映射字典。
    - category_map (dict): 从标签到类别名称的映射字典。
    
    Returns:
    - pd.DataFrame: 包含类别名和对应计数的数据框。
    """
    overall_counts = Counter()
    
    for folder in folder_paths:
        for root, dirs, files in os.walk(folder):
            for file in tqdm(files, desc=f"Processing {root}"):
                if file.endswith('.label'):  # 处理.label格式的文件
                    file_path = os.path.join(root, file)
                    labels = read_label_file(file_path, label_map)
                    counts = Counter(labels)
                    overall_counts.update(counts)

 
    
    # 将标签转换为类别名称
    category_counts = {category_map.get(label, "Unknown"): count
                       for label, count in overall_counts.items()}
    # #sequence08
    # category_counts['ignore'] = 22321839
    # category_counts['vegetation'] = 145937016
    # category_counts['pole'] = 1666427
    # category_counts['trunk'] = 5503279
    # category_counts['building'] = 56878900
    # category_counts['other-vehicle'] = 2222268
    # category_counts['traffic-sign'] = 381442
    # category_counts['car'] = 30856105
    # category_counts['sidewalk'] = 60280496
    # category_counts['terrain'] = 64111254    
    # category_counts['road'] = 87963626
    # category_counts['fence'] = 12642657
    # category_counts['motorcycle'] = 345868
    # category_counts['bicycle'] = 247796
    # category_counts['person'] = 476946   
    # category_counts['bicyclist'] = 306860
    # category_counts['parking'] = 5947692
    # category_counts['other-ground'] = 457650
    # category_counts['truck'] = 508704
    # category_counts['motorcyclist'] = 22737 
    
    #train
    category_counts['ignore'] = 73587231
    category_counts['vegetation'] = 627195745
    category_counts['pole'] = 6712285
    category_counts['trunk'] = 14189414
    category_counts['building'] = 311802516
    category_counts['other-vehicle'] = 5455888
    category_counts['traffic-sign'] = 1441988
    category_counts['car'] = 99431486
    category_counts['sidewalk'] = 338183720
    category_counts['terrain'] = 183603141    
    category_counts['road'] = 467085589
    category_counts['fence'] = 170001681
    category_counts['motorcycle'] = 936031
    category_counts['bicycle'] = 391766
    category_counts['person'] = 816813   
    category_counts['bicyclist'] = 298604
    category_counts['parking'] = 34577789
    category_counts['other-ground'] = 9173976
    category_counts['truck'] = 4586090
    category_counts['motorcyclist'] = 87779
 
    
    return pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])

def plot_label_distribution(label_counts):
    """
    绘制标签分布的条形图,并使用类别名称作为X轴标签。
    
    Args:
    - label_counts (pd.DataFrame): 包含类别名和对应计数的数据框。
    """ 
    
    
    label_counts.sort_values('Count', ascending=False, inplace=True)
    plt.rcParams['font.sans-serif'] = ['SimHei']#指定字体为SimHei
    plt.figure(figsize=(12, 8))
    plt.bar(label_counts['Category'], label_counts['Count'], color='skyblue')
    plt.yscale('log')  # 使用对数坐标轴
    # plt.xlabel('Category')
    # plt.ylabel('Count (log scale)')
    # plt.title('Distribution of Categories Across Files')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0411_count_category_all.png')
    
    plt.xlabel('类别',size=18)
    plt.ylabel('标签点数量 (log scale)',size=18)
    # plt.title('Distribution of Categories Across Files')
    plt.xticks(rotation=45, ha='right',size=18)
    plt.yticks(size=18)
    plt.tight_layout()
    plt.savefig('/home/ps/huichenchen/mmdetection3d/results2/0522_count_category_train.png')

# 配置路径和映射
# folder_paths = [ '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/00/labels', 
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/01/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/02/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/03/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/04/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/05/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/06/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/07/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/09/labels',
#                  '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/10/labels',]  # 实际的文件夹路径
# folder_paths = [ '/home/ps/huichenchen/mmdetection3d/data/semantickitti/sequences/04/labels',]  # 实际的文件夹路径
folder_paths = ['/home/ps/huichenchen/mmdetection3d/results2/test/0522']
label_map = {  # 标签ID到类别ID的映射
     0: 19,
    1: 19,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 19,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 19
}
category_map = {  # 类别ID到类别名称的映射
    0:'car',
    1:'bicycle', 
    2:'motorcycle',
    3:'truck',
    4:'other-vehicle',
    5:'person',
    6:'bicyclist', 
    7:'motorcyclist', 
    8:'road',
    9:'parking', 
    10:'sidewalk', 
    11:'other-ground',
    12:'building', 
    13:'fence', 
    14:'vegetation',
    15:'trunk',
    16:'terrain',
    17:'pole',
    18:'traffic-sign', 
    19:'ignore',
}

# 执行统计和绘图
label_counts = count_labels(folder_paths, label_map, category_map)
print(label_counts)  # 打印类别计数信息
plot_label_distribution(label_counts)









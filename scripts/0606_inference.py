import os
import glob
from tqdm import tqdm
import subprocess

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# 定义推理脚本的路径
demo_script = '/home/ps/huichenchen/mmdetection3d/demo/pcd_seg_demo.py'

# 定义模型配置文件和权重文件的路径
# config_file = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_multi-mix-lr01_2/cylinder3d_1xb2-multi-mix-lr01.py'
# checkpoint_file = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_multi-mix-lr01_2/epoch_33.pth'

# config_file = '/home/ps/huichenchen/mmdetection3d/results/cylinder3d/1xb2_turn3/cylinder3d_1xb2-3x_semantickitti.py'
# checkpoint_file = '/home/ps/huichenchen/mmdetection3d/results/cylinder3d/1xb2_turn3/epoch_35.pth'

config_file = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/0510-multi-instancemix_plusdistance_lr0008/cylinder3d_1xb2-multi-instancemix_plusdistance.py'
checkpoint_file = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/0510-multi-instancemix_plusdistance_lr0008/epoch_36.pth'
# 定义输入和输出文件夹的路径
input_folder = '/mnt/datasets/huichenchen/robosense/rs32_park/bin'
output_folder = '/mnt/datasets/huichenchen/robosense/rs32_park/multi-json2'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有.bin文件的路径
bin_files = glob.glob(os.path.join(input_folder, '*.bin'))

# 使用tqdm包装bin_files迭代器来显示进度条
for bin_file in tqdm(bin_files, desc='Running inference on bin files'):
    # 构建输出JSON文件的路径
    output_json_file = os.path.join(output_folder, os.path.basename(bin_file).replace('.bin', '.json'))
    
    # 构建命令行参数
    cmd = [
        'python', demo_script,
        bin_file, config_file, checkpoint_file,
        '--out-dir', os.path.dirname(output_json_file),
        
    ]
    
    # 使用subprocess运行命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 打印输出结果
    print(f"Inference result for {bin_file}:\n{result.stdout}")

print("Inference completed for all files.")
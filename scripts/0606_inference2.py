import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

cpu_cores = os.cpu_count()
max_workers = cpu_cores

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 定义推理脚本的路径
demo_script = '/home/ps/huichenchen/mmdetection3d/demo/pcd_seg_demo.py'

# 定义模型配置文件和权重文件的路径
config_file = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_multi-mix-lr01_2/cylinder3d_1xb2-multi-mix-lr01.py'
checkpoint_file = '/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_multi-mix-lr01_2/epoch_33.pth'

# 定义输入和输出文件夹的路径
input_folder = '/mnt/datasets/huichenchen/robosense/rs128/bin'
output_folder = '/mnt/datasets/huichenchen/robosense/rs128/test/test'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有.bin文件的路径
bin_files = glob.glob(os.path.join(input_folder, '*.bin'))

def run_inference(bin_file):
    # 构建输出JSON文件的路径
    output_json_file = os.path.join(output_folder, os.path.basename(bin_file).replace('.bin', '.json'))
    
    # 构建命令行参数
    cmd = [
        'python', demo_script,
        bin_file, config_file, checkpoint_file,
        '--out-dir', output_json_file,  # 注意这里需要输出目录的路径，而不是os.path.dirname(output_json_file)
    ]
    
    # 使用subprocess运行命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 返回结果
    return bin_file, result.stdout

# 使用ProcessPoolExecutor进行并行处理
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # 使用tqdm监控进度
    futures = [executor.submit(run_inference, bin_file) for bin_file in bin_files]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference on bin files"):
        bin_file, stdout = future.result()
        print(f"Inference result for {bin_file}:\n{stdout}")

print("Inference completed for all files.")
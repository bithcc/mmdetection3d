import os
import subprocess
import time

def run_command_on_bins(bin_folder, out_dir):
    # 设定你的其他路径和文件
    config_path = "/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_multi-mix-lr01_2/cylinder3d_1xb2-multi-mix-lr01.py"
    # config_path = "/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_turn5/cylinder3d_1xb2-3x_semantickitti.py"
    weight_path = "/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_multi-mix-lr01_2/epoch_33.pth"
    weight_path = "/home/ps/huichenchen/mmdetection3d/results2/cylinder3d/1xb2_turn5/epoch_36.pth"
    
    # 获取文件夹内所有.bin文件
    bin_files = [f for f in os.listdir(bin_folder) if f.endswith('.bin')]
    
    for bin_file in bin_files:
        bin_file_path = os.path.join(bin_folder, bin_file)
        
        # 构建命令
        command = f"CUDA_VISIBLE_DEVICES=6 python demo/pcd_seg_demo.py {bin_file_path} {config_path} {weight_path} --out-dir {out_dir}"
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行命令
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)
        
        # 计算并打印执行时间
        elapsed_time = time.time() - start_time
        print(f"Execution time for {bin_file}: {elapsed_time:.2f} seconds")

# 调用函数，指定bin文件所在的文件夹和输出目录
bin_folder = "/home/ps/huichenchen/mmdetection3d/results2/exp/bin_output/bin_robosense32-road"
output_directory = "/home/ps/huichenchen/mmdetection3d/results2/exp/json_output/test"
run_command_on_bins(bin_folder, output_directory)

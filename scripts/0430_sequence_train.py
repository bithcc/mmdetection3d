

import os
import sys
import time
#用于排队进行模型训练
cmd = 'CUDA_VISIBLE_DEVICES=3 /home/ps/miniconda3/envs/hcc_1/bin/python \
       /home/ps/huichenchen/mmdetection3d/tools/train.py \
       /home/ps/huichenchen/mmdetection3d/configs/cylinder3d/cylinder3d_1xb2-multistar-mix_plusdistance.py \
       --work-dir /home/ps/huichenchen/mmdetection3d/results2/cylinder3d/0430-multistar-mix_plusdistance' 

#0425-plusdistance --resume

def gpu_info(gpu_index=3):
    info = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])

    return power,memory


def narrow_setup(interval=1):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 1000 and gpu_power > 20 :  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()
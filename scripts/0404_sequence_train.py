import os
import sys
import time
#用于排队进行模型训练
cmd = '/home/ps/miniconda3/envs/hcc_ps/bin/python \
       /home/ps/huichenchen/2DPASS/main.py \
       --log_dir 2DPASS_semkitti --config config/2DPASS-semantickitti.yaml --gpu 5'
 \
      


def gpu_info(gpu_index=5):
    info = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])

    return power,memory


def narrow_setup(interval=10):
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
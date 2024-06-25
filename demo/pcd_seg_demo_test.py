# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import glob
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet3d.apis import LidarSeg3DInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('weights', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    call_args = vars(parser.parse_args())

    call_args['inputs'] = dict(points=call_args.pop('pcd'))

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args


def main():
    # 解析命令行参数
    init_args, call_args = parse_args()

    # 初始化推理器
    inferencer = LidarSeg3DInferencer(**init_args)
    
    # 直接修改文件夹名称
    new_input_folder = '/mnt/datasets/huichenchen/robosense/rs128/bin'
    call_args['inputs']['points'] = new_input_folder
    
    # 确保输出目录存在
    if not os.path.exists(call_args['out_dir']):
        os.makedirs(call_args['out_dir'])

    # 获取更新后的文件夹中所有.bin文件的路径
    bin_files = glob.glob(os.path.join(call_args['inputs']['points'], '*.bin'))

    # 遍历所有.bin文件并进行推理
    for bin_file in bin_files:
        # 更新call_args中的输入文件路径（这一步在这里是多余的，因为已经更新了）
        # call_args['inputs']['points'] = bin_file
        
        # 进行推理
        inferencer(**call_args)
        
        # 如果需要，可以在这里添加打印语句或其他逻辑

    if not (call_args['no_save_vis'] and call_args['no_save_pred']):
        print_log(
            f'Results have been saved at {call_args["out_dir"]}.',
            logger='current')

if __name__ == '__main__':
    main()

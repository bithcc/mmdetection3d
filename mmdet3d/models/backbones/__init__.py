# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt#@@@@所有可以使用的backbone

from .cylinder3d import Asymm3DSpconv
from .dgcnn import DGCNNBackbone
from .dla import DLANet
from .mink_resnet import MinkResNet
from .minkunet_backbone import MinkUNetBackbone
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG#@@@@这里可以替换吗？
from .second import SECOND
from .spvcnn_backone import MinkUNetBackboneV2, SPVCNNBackbone#@@@@这里可以进行替换吗？

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'DGCNNBackbone', 'PointNet2SASSG', 'PointNet2SAMSG',
    'MultiBackbone', 'DLANet', 'MinkResNet', 'Asymm3DSpconv',
    'MinkUNetBackbone', 'SPVCNNBackbone', 'MinkUNetBackboneV2'
]

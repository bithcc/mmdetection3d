# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt#@@@@所有可以使用的backbone

from .cylinder3d_parallel_withddcm import Parallel_withddcm_3DSpconv #5月23日 bithcc@@@@
from .cylinder3d_linear_withddcm import Linear_withddcm_3DSpconv #5月23日 bithcc@@@@
from .cylinder3d_multistar import Multistar_3DSpconv #4月30日 bithcc@@@@
from .cylinder3d_parallel_noddcm import Parallel_3DSpconv #4月22日 bithcc@@@@
from .cylinder3d_linear_noddcm import Linear_3DSpconv  #4月22日 bithcc@@@@
from .cylinder3d_test_noddcm import Basenoddcm_3DSpconv#4月15日 bithcc@@@@
from .cylinder3d_multinoddcm import Multinoddcm_3DSpconv#4月12日 bithcc@@@@
from .cylinder3d_multiplus import Multiplus_3DSpconv #4月10日 bithcc@@@@
from .cylinder3d_multilite import Multilite_3DSpconv #4月8日 bithcc@@@@
from .cylinder3d_multi import Multi_3DSpconv #4月8日 bithcc@@@@
from .cylinder3d_test import Base_3DSpconv #4月8日 bithcc@@@@
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
    'MinkUNetBackbone', 'SPVCNNBackbone', 'MinkUNetBackboneV2',
    'Base_3DSpconv','Multi_3DSpconv','Multilite_3DSpconv',#4月8日
    'Multiplus_3DSpconv','Multinoddcm_3DSpconv','Basenoddcm_3DSpconv',#后面的这些之前没写进去，有没有影响？
    'Linear_3DSpconv','Parallel_3DSpconv','Multistar_3DSpconv','Linear_withddcm_3DSpconv',
    'Parallel_withddcm_3DSpconv',
]

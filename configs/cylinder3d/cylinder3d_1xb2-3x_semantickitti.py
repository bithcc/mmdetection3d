_base_ = [
    'cylinder3d_4xb4-3x_semantickitti.py'
]

train_dataloader = dict(batch_size=2, )

grid_shape = [480, 360, 32]#&&&&原始点云的分割尺寸，维度上分别是radius（半径）、angle（角度）、height（高度）
model = dict(       #关注重点，这里是传入的模型的block的组成
    type='Cylinder3D', #检测器的名字
    data_preprocessor=dict(#@@@@这里就是对点云进行柱状体素化的模块
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],#x方向为0到50，向前为正，y方向为-pai到pai，z方向为-4到2，向上为正，需要根据数据集调整
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(#@@@@编码器模块，也是之后要修改的地方
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(#@@@@特征提取器模块，也是之后要修改的地方
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(#@@@@解码器模块，也是之后要修改的地方
        type='Cylinder3DHead',
        channels=128,
        num_classes=20,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)


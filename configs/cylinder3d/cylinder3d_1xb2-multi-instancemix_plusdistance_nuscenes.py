_base_ = [
    '../_base_/datasets/nus-3d.py', '../_base_/models/cylinder3d.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='NuScenesDataset'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='LaserInstanceMix',
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-25, 3],
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='NuScenesDataset'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
            [
                dict(
                    type='PolarMix',
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='NuScenesDataset'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
            [
                dict(
                    type='DistanceInstanceMix',
                    distance_bins=[10, 20, 30, 40,50],
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='NuScenesDataset'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
        ],
        prob=[0.33, 0.33,0.34]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points','gt_bboxes_3d' 'gt_labels_3d'])
]

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
        feat_channels=[64, 128, 256, 256],#加一层试试，2024年3月28日修改，尝试修改feature层数
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(#@@@@特征提取器模块，也是之后要修改的地方
        type='Multi_3DSpconv',
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
_base_ = [
   'cylinder3d_1xb2-multi-mix.py'
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
        dataset_type='semantickitti'),
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
                            dataset_type='semantickitti'),
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
                            dataset_type='semantickitti'),
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
                            dataset_type='semantickitti'),
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
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
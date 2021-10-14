_base_ =[
    '/opt/ml/detection/object-detection-level2-cv-01/configs/_base_/datasets/coco_detection.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/default_runtime.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/schedule_SGD.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/htc_Res/htc_without_semantic_r50_fpn_1x_coco.py'
]
img_norm_cfg = dict(
    mean=[0,0,0], std=[255.,255.,255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type = 'Mosaic'),
    #dict(type='Resize', img_scale=(300,300), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300,300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
            ),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP_50')
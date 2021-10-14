_base_ = [
    '/opt/ml/detection/object-detection-level2-cv-01/configs/_base_/datasets/coco_detection.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/_base_/schedules/schedule_1x.py', 
    '/opt/ml/detection/object-detection-level2-cv-01/configs/_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes =['General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']
img_norm_cfg = dict(
    mean=[0,0,0], std=[255.,255.,255.], to_rgb=True)

model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        #depth=50,
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='CTResNetNeck',
        #in_channel=2048,
        #num_deconv_filters=(1024,512,256),
        in_channel=512,
        num_deconv_filters = (256,128,64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=10,
        #in_channel=256,
        #feat_channel=256,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[0,0,0], std=[255,255,255], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    #dict(
    #    type='RandomCenterCropPad',
    #    crop_size=(512, 512),
    #    ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
    #    mean=[0, 0, 0],
    #    std=[1, 1, 1],
    #    to_rgb=True,
    #    test_pad_mode=None),
    dict(type='Resize', img_scale=(300,300), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize',img_scale=(300,300), keep_ratio=True),
            #dict(
            #    type='RandomCenterCropPad',
            #    ratios=None,
            #    border=None,
            #    mean=[0, 0, 0],
            #    std=[1, 1, 1],
            #    to_rgb=True,
            #    test_mode=True,
            #    test_pad_mode=['logical_or', 31],
            #    test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg','border')
                )
        ])
]


# Use RepeatDataset to speed up training

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train_kfold1.json',
            classes = classes,
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_kfold1.json',
        classes = classes,
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        classes = classes,
        img_prefix=data_root,
        pipeline=test_pipeline))

checkpoint_config = dict(interval=100)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
				## wandb 추가
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection'
               ))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[18, 24])  # the real step is [18*5, 24*5]
runner = dict(max_epochs=28)  # the real epoch is 28*5=140

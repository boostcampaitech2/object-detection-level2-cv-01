_base_ =[
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/dataset_Aug2.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/default_runtime.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/schedule_SGD.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/htc_Res/htc_without_semantic_r50_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

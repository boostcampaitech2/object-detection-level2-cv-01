_base_ = [
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/Cascade_RCNN_DetectoRS/cascade_rcnn_r50_fpn.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/dataset.py',
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/default_runtime.py', 
    '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/schedule_1x.py'
]
model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d'),
            style='pytorch')))
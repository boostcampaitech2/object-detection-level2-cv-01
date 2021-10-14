_base_ = [
    '/opt/ml/detection/mmdetection/configs/boostcamp/Cascade_RCNN_DetectoRS/cascade_rcnn_r50_fpn.py',
    '/opt/ml/detection/mmdetection/configs/boostcamp/dataset.py',
    '/opt/ml/detection/mmdetection/configs/boostcamp/schedule_1x.py', 
    '/opt/ml/detection/mmdetection/configs/boostcamp/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
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
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
runner = dict(max_epochs=40)

load_from = '/opt/ml/detection/mmdetection/work_dirs/Cascade_RCNN_DetectoRS/epoch_13.pth' #noqa

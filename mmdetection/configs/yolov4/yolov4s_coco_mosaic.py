model = dict(
    type='SingleStageDetector',
    backbone=dict(type='DarknetCSP', scale='v4s5p', out_indices=[3, 4, 5]),
    neck=dict(
        type='YOLOV4Neck',
        in_channels=[128, 256, 256],
        out_channels=[128, 256, 512],
        csp_repetition=1),
    bbox_head=dict(
        type='YOLOCSPHead', num_classes=80, in_channels=[128, 256, 512]),
    train_cfg=dict(),
    test_cfg=dict(
        min_bbox_size=0,
        nms_pre=-1,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)
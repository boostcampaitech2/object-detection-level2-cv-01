_base_= '/opt/ml/detection/object-detection-level2-cv-01/configs/boostcamp/Cascade_SwinT_PAFPN/Cascade_SwinT_PAFPN.py'
model = dict(
    rpn_head = dict(
        anchor_generator = dict(
            scales=[8,4]
        )
    )
)
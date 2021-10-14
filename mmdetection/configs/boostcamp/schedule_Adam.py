_base_ = '/opt/ml/detection/object-detection-level2-cv-01/configs/_base_/schedules/schedule_1x.py'
optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# optimizer = dict(_delete_ = True, type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1 / 10,
    min_lr=1e-6)
runner = dict(type='EpochBasedRunner', max_epochs=50)
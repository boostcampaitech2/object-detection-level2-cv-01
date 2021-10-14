# optimizer
optimizer = dict(type='AdamW', lr=0.00004, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1 / 10,
    min_lr=4e-6)
runner = dict(type='EpochBasedRunner', max_epochs=50)

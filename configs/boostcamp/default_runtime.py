checkpoint_config = dict(interval=1)

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

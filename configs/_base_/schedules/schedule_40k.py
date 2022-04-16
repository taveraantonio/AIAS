# optimizer
optimizer_config = dict()

# optimizer
optimizer = dict(type='AdamW',
                 lr=0.00006,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'pos_block': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                     'head': dict(lr_mult=10.)
                 }))

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=1e-7,
                 by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80_000)
checkpoint_config = dict(by_epoch=False, interval=10_000)
evaluation = dict(interval=40_000, metric='mIoU', pre_eval=True)

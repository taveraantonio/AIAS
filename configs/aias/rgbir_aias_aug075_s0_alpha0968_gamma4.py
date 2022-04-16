_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/segformer_mit-b5.py',
    # Dataset
    '../_base_/datasets/agrivision_rgbir.py',
    # Customization
    '../_base_/custom/aug_flip_rot90_jitter_075.py',
    # Training schedule
    '../_base_/schedules/schedule_40k.py'
]
# Random Seed
seed = 0
group = "segformer"

# optimizer
optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.00006,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'pos_block': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                     'head': dict(lr_mult=10.)
                 }))

lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=1e-7,
                 by_epoch=False)

data = dict(samples_per_gpu=2,
            workers_per_gpu=2,
            train=dict(sampling=dict(
                min_pixels=0,
                temp=0.1,
                minmax=True,
                alpha=0.968,
                gamma=4.0,
            )))
# important: for dynamic sampling also set return_confidence=True
model = dict(decode_head=dict(return_confidence=True))

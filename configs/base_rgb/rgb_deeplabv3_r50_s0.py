_base_ = [
    '../_base_/default_runtime.py',
    # Network Architecture
    '../_base_/models/deeplabv3_r50-d8.py',
    # Dataset
    '../_base_/datasets/agrivision_rgb.py',
    # Customization
    '../_base_/custom/base.py',
    # Training schedule
    '../_base_/schedules/schedule_40k.py'
]
# Random Seed
seed = 0
data = dict(samples_per_gpu=4, workers_per_gpu=2)

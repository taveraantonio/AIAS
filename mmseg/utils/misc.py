# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings

import torch


def find_latest_checkpoint(path, suffix='pth'):
    """This function is for finding the latest checkpoint.

    It will be used when automatically resume, modified from
    https://github.com/open-mmlab/mmdetection/blob/dev-v2.20.0/mmdet/utils/misc.py

    Args:
        path (str): The path to find checkpoints.
        suffix (str): File extension for the checkpoint. Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    """
    if not osp.exists(path):
        warnings.warn("The path of the checkpoints doesn't exist.")
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('The are no checkpoints in the path')
        return None
    latest = -1
    latest_path = ''
    for checkpoint in checkpoints:
        if len(checkpoint) < len(latest_path):
            continue
        # `count` is iteration number, as checkpoints are saved as
        # 'iter_xx.pth' or 'epoch_xx.pth' and xx is iteration number.
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def get_mean_std(img_metas: list, device: torch.device, batch_size: int, num_channels: int = 3):
    mean = [torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=device) for i in range(len(img_metas))]
    mean = torch.stack(mean).view(-1, num_channels, 1, 1)
    std = [torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=device) for i in range(len(img_metas))]
    std = torch.stack(std).view(-1, num_channels, 1, 1)
    repeat_factor = batch_size // len(img_metas)
    if repeat_factor > 1:
        mean = mean.repeat(repeat_factor, 1, 1, 1)
        std = std.repeat(repeat_factor, 1, 1, 1)
    return mean, std


def denorm(img, mean, std):
    """denormalizes the given tensor using matrix multiplication.
    """
    return img.mul(std).add(mean) / 255.0

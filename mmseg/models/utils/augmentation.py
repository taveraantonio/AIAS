from typing import Dict, Literal, Optional, cast

import torch
from kornia import augmentation as krn
from kornia.geometry.transform import warp_perspective
from pydantic import BaseModel
from torch import nn


class Temperature(BaseModel):
    start_value: float = 1.0
    end_value: float = 1.0
    max_decay_steps: int = 1


class AugParams(BaseModel):
    factor: float = 0.0
    loss: Literal["mse", "l1"] = "mse"
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    random_degrees: float = 360
    random_step: float = 90
    jitter_prob: float = 0.3
    jitter_strentgh: float = 0.15
    perspective_prob: float = 0.0
    perspective_dist: float = 0.1
    debug_augs: bool = True
    debug_interval: int = 1000
    temperature: Temperature = Temperature()


class CustomPerspective(krn.RandomPerspective):

    def apply_transform(self,
                        input: torch.Tensor,
                        params: Dict[str, torch.Tensor],
                        transform: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return warp_perspective(input,
                                transform, (height, width),
                                mode=self.flags["resample"].name.lower(),
                                align_corners=self.flags["align_corners"],
                                padding_mode="reflection")


class RepeatableTransform(nn.Module):

    def __init__(self, params: AugParams):
        super(RepeatableTransform, self).__init__()
        self.mean = None
        self.std = None
        self.degree_step = params.random_step
        self.jitter = krn.ColorJitter(brightness=params.jitter_strentgh,
                                      contrast=params.jitter_strentgh,
                                      saturation=params.jitter_strentgh,
                                      same_on_batch=False)
        self.rotate_linear = krn.RandomRotation(degrees=params.random_degrees,
                                                resample="bilinear",
                                                align_corners=True,
                                                p=1.0)
        self.rotate_nearest = krn.RandomRotation(degrees=params.random_degrees,
                                                 resample="nearest",
                                                 align_corners=True,
                                                 p=1.0)
        self.perspective_linear = CustomPerspective(distortion_scale=params.perspective_dist,
                                                    resample="bilinear",
                                                    align_corners=True,
                                                    p=params.perspective_prob)
        self.perspective_nearest = CustomPerspective(distortion_scale=params.perspective_dist,
                                                     resample="nearest",
                                                     align_corners=True,
                                                     p=params.perspective_prob)
        self.hflip = krn.RandomHorizontalFlip(p=params.hflip_prob)
        self.vflip = krn.RandomVerticalFlip(p=params.vflip_prob)

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def compute_params(self, batch_shape: torch.Size, device: torch.device):
        rotate_params = self.rotate_linear.forward_parameters(batch_shape)
        if self.degree_step > 1:
            rotation = (rotate_params['degrees'] / self.degree_step).int()
            rotate_params['degrees'] = (rotation * self.degree_step).float()
        params = {
            "jitter": self.jitter.forward_parameters(batch_shape),
            "hflip": self.hflip.forward_parameters(batch_shape),
            "vflip": self.vflip.forward_parameters(batch_shape),
            "perspective": self.perspective_linear.forward_parameters(batch_shape),
            "rotate": rotate_params
        }
        return set_device_recursive(params, device=device)

    def _adapt_params(self, params: dict, new_shape: tuple, device: torch.device):
        # when the shape changes, absolute values need to be scaled to a different height and width
        # features are usually 128x128, while images are 512x512
        params["hflip"]["batch_shape"] = torch.tensor(new_shape, device=device)
        params["vflip"]["batch_shape"] = torch.tensor(new_shape, device=device)
        old_dims = params["perspective"]["forward_input_shape"][2:]
        new_dims = torch.tensor(new_shape[2:], device=device)
        start = params["perspective"]["start_points"]
        end = params["perspective"]["end_points"]
        params["perspective"]["start_points"] = start / old_dims * new_dims
        params["perspective"]["end_points"] = end / old_dims * new_dims
        params["perspective"]["forward_input_shape"] = torch.tensor(new_shape, device=device)
        return params

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor = None,
                params: dict = None,
                custom_shape: tuple = None,
                color_transform: bool = True) -> torch.Tensor:
        # inputs can be image or features, the shape must be overridden
        if custom_shape is not None:
            params = self._adapt_params(params, custom_shape, inputs.device)
        # 1. apply color only in image
        # 2. apply geometric tranform on both
        out = self.vflip(self.hflip(inputs, params.get("hflip")), params.get("vflip"))
        out = self.rotate_linear(out, params.get("rotate"))
        out = self.perspective_linear(out, params.get("perspective"))
        if color_transform:
            # denormalize, apply jitter, renormalize
            out.mul_(self.std).add_(self.mean).div_(255.0)
            out_rgb = out[:, :3]
            out_rgb = self.jitter(out_rgb, params.get("jitter"))
            out[:, :3] = out_rgb
            out.mul_(255.0).sub_(self.mean).div_(self.std)
        # 3. infer geometry params to mask, kornia does not work with longs
        if mask is not None:
            mask = mask.float()
            mask_out = self.vflip(self.hflip(mask, params.get("hflip")), params.get("vflip"))
            mask_out = self.rotate_nearest(mask_out, params.get("rotate"))
            mask_out = self.perspective_nearest(mask_out, params.get("perspective"))
            return out, mask_out.long()
        return out


def set_device_recursive(params: dict, device):
    for k, v in params.items():
        if isinstance(v, dict):
            set_device_recursive(v, device)
        if isinstance(v, torch.Tensor):
            params[k] = v.to(device)
    return params

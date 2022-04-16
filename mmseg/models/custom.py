import os
from typing import Iterable, List

import mmcv
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as fn

from mmseg.datasets.sampling import SamplingDataset
from mmseg.models.builder import MODELS
from mmseg.models.utils.augmentation import AugParams, RepeatableTransform
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.wrapper import SegmentorWrapper
from mmseg.utils.misc import denorm, get_mean_std


@MODELS.register_module()
class CustomModel(SegmentorWrapper):

    def __init__(self,
                 model_config: dict,
                 max_iters: int,
                 resume_iters: int,
                 num_channels: int = 3,
                 aug: dict = None,
                 **kwargs):
        super().__init__(model_config, max_iters, resume_iters, num_channels, **kwargs)
        aug = aug or {}
        self.dataset = None
        self.aug_params = AugParams(**aug)
        self.augment = self.aug_params.factor > 0
        self.aug_transform = RepeatableTransform(self.aug_params) if self.augment else None
        mmcv.print_log(f"Using augmentation invariance: {self.augment}")
        mmcv.print_log("Augmentation parameters:")
        for k, v in self.aug_params.dict().items():
            mmcv.print_log(f"{k:<20s}: {str(v)}")
        # create loss module
        if self.aug_params.loss == "mse":
            self.aug_loss = nn.MSELoss(reduction="none")
        elif self.aug_params.loss == "l1":
            self.aug_loss = nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss: '{self.aug.loss}'")

    def link_dataset(self, dataset: SamplingDataset):
        """Simply stores a pointer to the current training set.

        Args:
            dataset (SamplingDataset): Dataset with dynamic sampling
        """
        self.dataset = dataset

    def compute_aug_loss(self,
                         features: torch.Tensor,
                         mask: torch.Tensor,
                         aug_params: dict,
                         split_at: int,
                         prefix: str = "",
                         debug: bool = False) -> dict:
        """Computes the augmentation equivariance loss, intended as squared L2 norm between:
        T(f(x)) <=> f(T(x)), where T indicates augmentations, f(x) the pixel embeddings, namely
        the features of the model prior to the class convolutions (e.g., a tensor [batch, 768, 128, 128]).

        Args:
            features (torch.Tensor): feature tensor with size (b, channels, h, w)
            mask (torch.Tensor): binary mask to exclude ignored areas (1 where valid, 0 where ignored)
            aug_params (dict): parameters used on the input, necessary to augment the features
            split_at (int): index on the batch that divides original and augmented features
            prefix (str, optional): prefix for the output dict. Defaults to "".
            debug (bool, optional): whether to return extra stuff or not. Defaults to False.

        Returns:
            Dict[str, Any]: dictionary with additional loss and optional stuff
        """
        # first part of the batch are the original images, the last the augmented
        f_x = features[:split_at]
        f_tx = features[split_at:]
        t_fx = self.aug_transform(f_x, params=aug_params, custom_shape=f_x.shape, color_transform=False)
        # compute the loss for each "pixel" embedding at 12x128
        # mask to exclude areas set to ignore_index, multiply pixelwise
        # then reduce and multiply for a reduction factor (decreases impact on CE)
        mse_pixelwise = self.aug_loss(t_fx, f_tx)
        _, _, h, w = mse_pixelwise.shape
        mask = fn.interpolate(mask, size=(h, w), mode="nearest")
        mse_pixelwise = mse_pixelwise * mask
        loss_inv = mse_pixelwise.mean() * self.aug_params.factor
        result = {f"{prefix}.loss_aug": loss_inv}
        if debug:
            result["debug"] = (f_tx, t_fx, mask, mse_pixelwise.detach().mean(axis=1))
        return result

    def vectorize(self, values: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Transforms the given dictionary of index: value into a tensor of size num_classes,
        where each position i contains the value of the corresponding index, if present, 0 otherwise.

        Args:
            values (dict): dictionary of <index: value> pairs
            device (torch.device): device to store the tensor on
            dtype (torch.dtype): type of the torch tensor

        Returns:
            torch.Tensor: torch tensor of size num_classes
        """
        array = torch.zeros(self.num_classes, device=device, dtype=dtype)
        for i, v in values.items():
            array[i] = v
        return array

    def compute_classwise_confidence(self, confidence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Reduces the given confidence tensor into classwise values, a vector of length num_classes.

        Args:
            confidence (torch.Tensor): softmaxed probs, shape [batch, num_classes, h, w]
            labels (torch.Tensor): index labels, shape [batch, 1, h, w]

        Returns:
            torch.Tensor: classwise confidence and classwise pixel count on the current batch
        """
        classes, counts = torch.unique(labels, return_counts=True)
        class_counts = dict(zip(classes.tolist(), counts))
        class_counts.pop(255, None)
        classwise_confidence = dict()
        for category in class_counts.keys():
            # [ n, 512, 512], label is [n, 1, 512, 512]
            channel_conf = confidence[:, category]
            classwise_conf = channel_conf[labels.squeeze(1) == category].mean()
            classwise_confidence[category] = classwise_conf
        return classwise_confidence

    def check_keys(self, data: dict, expected: Iterable[str]) -> str:
        """Checks whether one of the given list of keys belongs to the data dictionary.

        Args:
            data (dict): dict with a bunch of keys
            expected (Iterable[str]): list of keys that could appear in the dict.

        Returns:
            str: the first key that appears, None otherwise.
        """
        for key in expected:
            if key in data:
                return key
        return None

    def forward_train(self, img: torch.Tensor, img_metas: List[dict], gt_semantic_seg: torch.Tensor):
        # mmcv.print_log(f"iter: {self.local_iter}")
        # mmcv.print_log(f"input shape: {img.shape}")
        # mmcv.print_log(f"label shape: {gt_semantic_seg.shape}")
        batch_size = img.shape[0]
        device = img.device
        debug_iter = self.aug_params.debug_augs and self.local_iter % self.aug_params.debug_interval == 0

        # prepare mean and std for the augmentations
        means, stds = get_mean_std(img_metas=img_metas,
                                   device=device,
                                   batch_size=batch_size,
                                   num_channels=self.num_channels)
        if self.aug_transform is not None:
            self.aug_transform.set_stats(means, stds)

        # INVARIANCE TRANSFORM - source, step 1
        # # compute x and T(x), then concatenate in a single batch
        if self.augment:
            aug_params = self.aug_transform.compute_params(img.shape, device=device)
            img_aug, gt_aug = self.aug_transform(img, gt_semantic_seg, params=aug_params)
            img = torch.cat((img, img_aug), dim=0)
            gt_semantic_seg = torch.cat((gt_semantic_seg, gt_aug), dim=0)

        # Forward on the (possibly augmented) batch with standard segmentation
        losses = super().forward_train(img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=self.augment)

        # check if it returned confidence: if so, update dataset stats
        confidence_key = self.check_keys(losses, expected=("decode.confidence", "decode_1.confidence"))
        if confidence_key:
            confidence = self.compute_classwise_confidence(losses[confidence_key], gt_semantic_seg)
            self.dataset.update_statistics(confidence, iters=self.local_iter)

        # continue with augmentation invariance
        aug_losses = dict()
        if self.augment:
            # create empty dict to temporarily store aug outs
            for key, features in losses.items():
                if not key.endswith("features"):
                    continue
                prefix, _ = key.split(".")
                mask = (gt_aug != 255).type(torch.uint8)
                aug_out: dict = self.compute_aug_loss(features=features,
                                                      mask=mask,
                                                      aug_params=aug_params,
                                                      split_at=batch_size,
                                                      prefix=prefix,
                                                      debug=debug_iter)
                # remove debug stuff from the dict and update the group dict
                debug_vars = aug_out.pop("debug", None)
                aug_losses.update(aug_out)
                # debug plots at each interval, only if required
                if debug_iter:
                    f_tx, t_fx, mask, mse = debug_vars
                    self._plot_aug_debug(img[:batch_size],
                                         img[batch_size:],
                                         src_labels=gt_semantic_seg[:batch_size],
                                         aug_labels=gt_semantic_seg[batch_size:],
                                         src_features=t_fx,
                                         aug_features=f_tx,
                                         mask=mask,
                                         loss=mse,
                                         means=means,
                                         stds=stds,
                                         batch_size=batch_size,
                                         prefix=prefix)
                    # also, print the confidence-based dynamic weights
                    if self.dataset:
                        probs = self.dataset.compute_probs()
                        confs = self.dataset.conf_buffer.get_counts()
                        mmcv.print_log(f'Current probs:  {probs}', 'mmseg')
                        mmcv.print_log(f"classwise conf: {confs}", 'mmseg')

        # update the global dictionary of losses with the list of augmentation losses
        losses.update(aug_losses)
        # increment iteration
        self.local_iter += 1
        return losses

    def _plot_aug_debug(self,
                        src_img: torch.Tensor,
                        aug_img: torch.Tensor,
                        src_labels: torch.Tensor,
                        aug_labels: torch.Tensor,
                        src_features: torch.Tensor,
                        aug_features: torch.Tensor,
                        mask: torch.Tensor,
                        loss: torch.Tensor,
                        means: torch.Tensor,
                        stds: torch.Tensor,
                        batch_size: int,
                        prefix=""):
        # create dir in the cwd to store plots
        out_dir = os.path.join(self.work_dir, 'aug_debug')
        os.makedirs(out_dir, exist_ok=True)
        # denormalize images to restore 0-1 range
        vis_src_img = torch.clamp(denorm(src_img, means[:batch_size], stds[:batch_size]), 0, 1)
        vis_aug_img = torch.clamp(denorm(aug_img, means[:batch_size], stds[:batch_size]), 0, 1)
        # iterate batch and save plots
        for i in range(batch_size):
            index = np.random.randint(0, src_features.shape[1])
            rows, cols = 2, 4
            fig, axs = plt.subplots(rows,
                                    cols,
                                    figsize=(3 * cols, 3 * rows),
                                    gridspec_kw={
                                        'hspace': 0.1,
                                        'wspace': 0,
                                        'top': 0.95,
                                        'bottom': 0,
                                        'right': 1,
                                        'left': 0
                                    })
            subplotimg(axs[0][0], vis_src_img[i, :3], 'Src. image')
            subplotimg(axs[1][0], vis_aug_img[i, :3], 'Aug. image')

            subplotimg(axs[0][1], src_labels[i], 'Src. Label', cmap='agrivision')
            subplotimg(axs[1][1], aug_labels[i], 'Aug. Label', cmap='agrivision')

            subplotimg(axs[0][2], mask[i], 'Aug. mask', cmap='gray', vmin=0, vmax=1)
            subplotimg(axs[1][2], loss[i], 'Aug. loss', cmap="plasma")

            subplotimg(axs[1][3], aug_features[i, index], 'f(T(x))', cmap="viridis")
            subplotimg(axs[0][3], src_features[i, index], 'T(f(x))', cmap="viridis")

            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(os.path.join(out_dir, f'{prefix}_{(self.local_iter + 1):06d}_{i}.png'))
            plt.close(fig)

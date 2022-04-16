import json
import os.path as osp
from abc import abstractmethod
from typing import Dict

import mmcv
import numpy as np
import torch

from mmseg.datasets.buffer import FixedBuffer
from mmseg.datasets.custom import CustomDataset


def get_class_freqs(data_root: str, temperature: float, minmax: bool = False):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {k: v for k, v in sorted(overall_class_stats.items(), key=lambda item: item[0])}
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    if minmax:
        freq = (freq - freq.min() + 0.005) / (freq.max() - freq.min() + 0.005)
        freq = freq/freq.sum()
    else:
        freq = torch.softmax(freq, dim=-1)
    return list(overall_class_stats.keys()), freq.numpy()


def softmax(x: np.ndarray):
    """Quick softmax function for numpy arrays, instead of converting to torch
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SamplingDataset(CustomDataset):

    def __init__(self, sampling: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.sampling_enabled = sampling is not None
        if self.sampling_enabled:
            self.min_pixels = sampling['min_pixels']
            self.gamma = sampling.get("gamma", 1)
            # memory buffer holding a fixed amount of class-wise pixel counts
            alpha = sampling["alpha"]
            self.conf_buffer = FixedBuffer(num_classes=len(self.CLASSES), alpha=alpha)

            self.class_list, self.class_freq = get_class_freqs(self.data_root, sampling["temp"],
                                                               sampling.get("minmax", False))
            mmcv.print_log(f'Classes            : {self.class_list}', 'mmseg')
            mmcv.print_log(f'Normalized weights.: {self.class_freq}', 'mmseg')
            mmcv.print_log(f'minmax             : {sampling["minmax"]}', 'mmseg')

            with open(osp.join(self.data_root, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items() if int(k) in self.class_list
            }
            self.samples_with_class = {}
            for c in self.class_list:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.img_infos):
                file = dic['ann']['seg_map']
                self.file_to_idx[file] = i

    def compute_probs(self):
        average_class_confidence = self.conf_buffer.get_counts()
        weighted_class_confidence = 1 - self.class_freq * average_class_confidence
        weighted_class_confidence = weighted_class_confidence**self.gamma
        weighted_class_confidence = (weighted_class_confidence - weighted_class_confidence.min() + 0.005) / (weighted_class_confidence.max() - weighted_class_confidence.min() + 0.005)
        weighted_class_confidence = weighted_class_confidence / weighted_class_confidence.sum()
        # weighted_class_confidence = softmax(weighted_class_confidence)
        return weighted_class_confidence

    def get_rare_class_sample(self):
        # compute weights for random sampling
        weighted_class_confidence = self.compute_probs()

        # UNCOMMENT THE FOLLOWING LINE(S) TO CHECK
        c = np.random.choice(self.class_list, p=weighted_class_confidence)
        # mmcv.print_log(f'weights:      {weighted_class_confidence}', 'mmseg')
        # mmcv.print_log(f'class chosen: {c}', 'mmseg')

        f = np.random.choice(self.samples_with_class[c])
        idx = self.file_to_idx[f]
        sample = self.prepare_batch(idx)
        return sample

    def update_statistics(self, class_confidence: Dict[int, float], iters: int):
        # mmcv.print_log(f'batch:   {class_confidence}', 'mmseg')
        self.conf_buffer.append(class_confidence, iters=iters)

    @abstractmethod
    def prepare_batch(self, idx: int):
        raise NotImplementedError()

    def prepare_train_img(self, idx: int):
        """Yet another wrapper to switch between RCS and standard random sampling.
        """
        if self.sampling_enabled:
            return self.get_rare_class_sample()
        return self.prepare_batch(idx)

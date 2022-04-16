import os

from .builder import DATASETS
from .sampling import SamplingDataset


@DATASETS.register_module()
class AgricultureVisionDataset(SamplingDataset):

    CLASSES = ("background", "double_plant", "drydown", "endrow", "nutrient_deficiency", "planter_skip", "water",
               "waterway", "weed_cluster")
    PALETTE = [[64, 64, 64], [23, 190, 207], [32, 119, 180], [148, 103, 189], [43, 160, 44], [127, 127, 127],
               [214, 39, 40], [140, 86, 75], [255, 127, 14]]

    def __init__(self, nir_dir: str = None, **kwargs):
        super().__init__(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)
        self.nir_dir = os.path.join(self.data_root, nir_dir)
        self.nir_infos = self.load_annotations(self.nir_dir, self.img_suffix, None, None, None)

    def pre_pipeline(self, results: dict):
        super().pre_pipeline(results)
        results["nir_prefix"] = self.nir_dir

    def prepare_batch(self, idx: int):
        img_info = self.img_infos[idx]
        nir_info = self.nir_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, nir_info=nir_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx: int):
        img_info = self.img_infos[idx]
        nir_info = self.nir_infos[idx]
        ann_info = self.img_infos[idx].get('ann')
        results = dict(img_info=img_info, nir_info=nir_info)
        if ann_info is not None:
            results["ann_info"] = ann_info
        self.pre_pipeline(results)
        return self.pipeline(results)
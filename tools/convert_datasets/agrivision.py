# This script just computes class-wise statistics over the already preprocessed dataset.
# It expects the following directory structure:
#   train
#   |-- gt
#   |-- images
#   |   |-- rgb
#   |   |-- nir
#   valid
#   |...
#   test
#   |...
# Of course, the stats are computed on the training set only.
import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
# constant vars that depends on LoveDA
# the ignore index is set to 0 and not 255
SUBSET_DIRS = ("train", "valid", "test")
LABEL_DIR = "gt"
IGNORE_INDEX = 255


def compute_stats(path: Path) -> dict:
    # read image, count index occurences
    # remove any ignore_index counts if present
    image = np.asarray(Image.open(str(path)))
    labels, counts = np.unique(image, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(labels, counts)}
    class_counts.pop(IGNORE_INDEX, None)
    class_counts["file"] = str(path)
    return class_counts


def save_class_stats(dst_dir: Path, sample_class_stats: List[dict]):
    """Derived from DAformer
    """
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(str(dst_dir / 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(str(dst_dir / 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(str(dst_dir / 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Computes stats for the Agriculture-Vision dataset")
    parser.add_argument("-p", "--path", required=True, default="data/agrivision", help="Dataset folder path")
    parser.add_argument("-o", "--out_dir", default="data/agrivision", type=str, help="output path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    LOG.info("Arguments:")
    for k, v in vars(args).items():
        LOG.info(f"{k:<20s}: {str(v)}")

    dataset_path = Path(args.path)
    out_dir = Path(args.out_dir)

    # create an empty list to gather label statistics
    dataset_stats = []
    label_files = list(glob(str(dataset_path / SUBSET_DIRS[0] / LABEL_DIR / "*.png")))
    LOG.info("Labels found: %d", len(label_files))

    for file_path in tqdm(label_files):
        # just need the stats from the train set. Nones will be filtered out
        image_stats = compute_stats(file_path)
        dataset_stats.append(image_stats)

    LOG.info("Storing statistics for the %s domain...")
    save_class_stats(out_dir, sample_class_stats=dataset_stats)
    LOG.info("Done!")


if __name__ == "__main__":
    main()

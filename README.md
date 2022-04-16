# Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images

This is the official PyTorch implementation of our work: "Augmentation Invariance and Adaptive Sampling in Semantic 
Segmentation of Agricultural Aerial Images" published at the "3rd International Workshop and Prize Challenge on
Agriculture-Vision: Challenges and Opportunities for Computer Vision in Agriculture" in conjunction with the "IEEE/CVF 
Computer Vision and Pattern Recognition Conference (CVPR 2022)".

<br>

In this paper, we investigate the problem of Semantic Segmentation for agricultural aerial imagery. We observe that 
the existing methods used for this task are designed without considering two characteristics of  the aerial data: (i) 
the top-down perspective implies 
that the model cannot rely on a fixed semantic structure of the scene, because the same scene may be experienced with 
different rotations of the sensor; (ii) there can be a strong  imbalance in the distribution of semantic classes because 
the relevant objects of the scene may appear at extremely different scales (e.g., a field of crops and a small vehicle).
We propose a solution to these problems based on two ideas: (i) we use together a set of suitable augmentation and a 
consistency loss to guide the model to learn semantic representations that are invariant to the photometric and geometric 
shifts typical of the top-down perspective (Augmentation Invariance); (ii) we use a sampling method (Adaptive Sampling)
that select the training images based on a measure of pixel-wise distribution of classes and actual network confidence. 

## Installation

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install -e .
```

## Training

### Single GPU
```bash
 CUDA_VISIBLE_DEVICES=$DEVICES tools/train.py [CONFIG_FILE] [OPTIONS]
```

### Multi GPU
```bash
 source scripts/dist_train.sh [GPUS, e.g. 0,1] [CONFIG_FILE] [OPTIONS]
```
#### Example
```bash
 source scripts/dist_train.sh 0,1 configs/aias/rgb_aias_aug075_s0_alpha0968_gamma4.py
 source scripts/dist_train.sh 2,3 configs/aias/rgbir_aias_aug075_s0_alpha0968_gamma4.py
```


## Testing
```bash
 CUDA_VISIBLE_DEVICES=$DEVICES python tools/test.py [CONFIG] [CHECKPOINT] --eval mIoU
```


## Cite us
If you use this repository, please consider to cite us:

    @InProceedings{Tavera_2022_CVPR,
    author    = {Tavera, Antonio and Arnaudo, Edoardo and Masone, Carlo and Caputo, Barbara},
    title     = {Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {},
    year      = {2022},
    pages     = {}}

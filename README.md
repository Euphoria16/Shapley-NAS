## Shapley-NAS

This is the official pytorch implementation for the paper: [*Shapley-NAS: Discovering Operation Contribution for Neural Architecture Search*](https://openaccess.thecvf.com/content/CVPR2022/html/Xiao_Shapley-NAS_Discovering_Operation_Contribution_for_Neural_Architecture_Search_CVPR_2022_paper.html), 
which is accepted by CVPR2022. This repo contains the implementation of architecture search and evaluation on CIFAR-10 and ImageNet using our proposed Shapley-NAS.

![intro](https://github.com/Euphoria16/Shapley-NAS/blob/main/figs/Shapley-NAS.png)

## Quick Start

### Prerequisites

- python>=3.5
- pytorch>=1.1.0
- torchvision>=0.3.0 



## Usage

### Architecture Search on CIFAR-10

To search CNN cells on CIFAR-10, please run
```
export CUDA_VISIBLE_DEVICES=0
python -W ignore train_search.py    \
--batch_size 256    \
--shapley_momentum 0.8  \
--save cifar10_shapley    \
--data /path/to/cifar10
```
or simply use the command
```
bash run/search_cifar10_shapley.sh
```
 
### Architecture Search on ImageNet
To search CNN cells on ImageNet, please run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore train_search_imagenet.py    \
--batch_size 1024    \
--save imagenet_shapley    \
--shapley_momentum 0.8  \
--data /path/to/imagennet  \
```
or simply use the command
```
bash run/search_imagenet_shapley.sh
```


### Architecture Evaluation on CIFAR-10
To evaluate the derived architecture on CIFAR-10, please run
```
export CUDA_VISIBLE_DEVICES=0
python -W ignore train.py   \
--data /path/to/cifar10 \
--save train_cifar10   \
--auxiliary \
--cutout    \

```
or simply use the command
```
bash run/train_cifar10.sh
```

### Architecture Evaluation on ImageNet
To evaluate the derived architecture on ImageNet, please run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore train_imagenet.py \
 --tmp_data_dir /path/to/imagenet \
 --save train_imagenet \
 --workers 16   \
 --auxiliary \
 --note imagenet_shapley    \
```
or simply use the command
```
bash run/train_imagenet.sh
```

## Citation

Please cite our paper if you find it useful in your research:
```
@InProceedings{Xiao_2022_CVPR,
    author    = {Xiao, Han and Wang, Ziwei and Zhu, Zheng and Zhou, Jie and Lu, Jiwen},
    title     = {Shapley-NAS: Discovering Operation Contribution for Neural Architecture Search},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}
}
```

## Acknowledgements

We thank the authors of following works for opening source their excellent codes.

- [DARTS](https://github.com/quark0/darts)
- [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS)
- [DARTS-PT](https://github.com/ruocwang/darts-pt)
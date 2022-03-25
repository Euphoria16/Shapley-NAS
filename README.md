## Shapley-NAS: Discovering Operation Contribution for Neural Architecture Search

This is the pytorch implementation for the paper: *Shapley-NAS: Discovering Operation Contribution for Neural Architecture Search*, 
which is accepted by CVPR2022. This repo contains the implementation of architecture search and evaluation on CIFAR-10 and ImageNet using our proposed Shapley-NAS.


## Quick Start

### Prerequisites

- python>=3.5
- pytorch>=1.1.0
- torchvision>=0.3.0 




### Architecture Search on CIFAR-10

```
export CUDA_VISIBLE_DEVICES=0
python -W ignore train_search.py    \
--batch_size 256    \
--shapley_momentum 0.8  \
--save cifar10_shapley    \
--data cifar10_data
```
 
### Architecture Search on ImageNet

```
export CUDA_VISIBLE_DEVICES=0,1
python -W ignore train_search_imagenet.py    \
--batch_size 1024    \
--save imagenet_shapley    \
--shapley_momentum 0.8  \
--data /ILSVRC2012  \
```
### Architecture Evaluation on CIFAR-10

```
export CUDA_VISIBLE_DEVICES=0
python -W ignore train.py   \
--data cifar10_data \
--save train_cifar10   \
--auxiliary \
--cutout    \

```

### Architecture Evaluation on ImageNet

```
export CUDA_VISIBLE_DEVICES=0,1
python -W ignore train_imagenet.py \
 --tmp_data_dir /ILSVRC2012 \
 --save train_imagenet \
 --workers 16   \
 --auxiliary \
 --note imagenet_shapley    \
```
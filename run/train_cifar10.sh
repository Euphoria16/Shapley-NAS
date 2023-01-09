export CUDA_VISIBLE_DEVICES=0
python -W ignore train.py   \
--data ~/Datasets/cifar-10-batches-py/ \
--save train_cifar10   \
--auxiliary \
--cutout
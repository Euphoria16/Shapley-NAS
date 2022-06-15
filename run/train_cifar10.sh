export CUDA_VISIBLE_DEVICES=0
python -W ignore train.py   \
--data /path/to/cifar10 \
--save train_cifar10   \
--auxiliary \
--cutout    \
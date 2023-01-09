export CUDA_VISIBLE_DEVICES=0
python -W ignore train_search.py    \
--batch_size 256    \
--shapley_momentum 0.8  \
--save cifar10_shapley    \
--data ~/Datasets/cifar-10-batches-py/
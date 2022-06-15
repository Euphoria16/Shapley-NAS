export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore train_search_imagenet.py    \
--batch_size 1024    \
--save imagenet_shapley    \
--shapley_momentum 0.8  \
--data /path/to/imagennet  \
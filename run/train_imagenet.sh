export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore train_imagenet.py \
 --tmp_data_dir "/mnt/hdd0/hjyoo/Datasets/imagenet/" \
 --save train_imagenet \
 --workers 16   \
 --auxiliary \
 --note imagenet_shapley    \
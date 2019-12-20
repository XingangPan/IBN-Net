#!/usr/bin/env sh
model=resnet50_ibn_a_old
data_path=/pathToYourImageNetDataset/

python -u imagenet.py \
    -a $model \
    --test-batch 100 \
    --model_weight pretrained/${model}.pth \
    -e \
    -j 16 \
    --data $data_path \
    --gpu_id 0,1 \
    $* 2>&1|tee pretrained/${model}.log &

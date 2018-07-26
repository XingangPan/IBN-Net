#!/usr/bin/env sh
model=resnet50_ibn_a
data_path=/pathToYourImageNetDataset/
EXP_DIR=exp/$model
mkdir -p $EXP_DIR

python -u imagenet.py \
    -a $model \
    -j 32 \
    --data $data_path \
    --train-batch 256 \
    --test-batch 100 \
    --lr 0.1 \
    --epochs 100 \
    -c exp/${model} \
    --gpu_id 0,1,2,3,4,5,6,7 \
    2>&1 | tee exp/${model}/log.txt

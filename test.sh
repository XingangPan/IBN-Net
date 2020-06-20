#!/usr/bin/env sh

MODEL=resnet50_ibn_a
DATA_PATH=/pathToYourImageNetDataset/

python -u imagenet.py \
    -a $MODEL \
    --data $DATA_PATH \
    --pretrained \
    --test-batch 100 \
    -e \
    -j 16 \
    --gpu_id 0,1

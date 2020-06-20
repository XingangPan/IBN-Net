#!/usr/bin/env sh

MODEL=resnet50_ibn_a
DATA_PATH=/pathToYourImageNetDataset/
EXP_DIR=exp/$MODEL
mkdir -p $EXP_DIR

python -u imagenet.py \
    -a $MODEL \
    -j 32 \
    --data $DATA_PATH \
    --train-batch 256 \
    --test-batch 100 \
    --lr 0.1 \
    --epochs 100 \
    -c exp/${MODEL} \
    --gpu_id 0,1,2,3,4,5,6,7 \
    2>&1 | tee exp/${MODEL}/log.txt

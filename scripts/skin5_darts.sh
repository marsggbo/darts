#!/bin/bash

export PYTHONPATH="$(pwd)"

python ./cnn/train_search.py \
    --data_path="../enas_origin/data/skin5" \
    --data_type=0 \
    --image_size=224 \
    --batch_size=2 \
    --output_classes=5 \
    --learning_rate=0.025 \
    --learning_rate_min=0.0001 \
    --momentum=0.9 \
    --weight_decay=3e-4 \
    --report_freq=200 \
    --gpu=0 \
    --epochs=200 \
    --init_channels=16 \
    --layers=8 \
    --model_path="./data/models/skin5.pt" \
    --cutout \
    --cutout_length=16 \
    --drop_path_prob=0.3 \
    --seed=2 \
    --grad_clip=5 \
    --train_portion=0.5 \
    --arch_learning_rate=3e-4 \
    --arch_weight_decay=1e-3 \
    "$@"

    
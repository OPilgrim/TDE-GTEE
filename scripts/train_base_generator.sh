#!/bin/bash

python train_gtee.py \
    --model=gen \
    --ckpt_name=gen-ace2 \
    --dataset=ACE \
    --train_file=data/ACE05-E/train.oneie.json \
    --val_file=data/ACE05-E/dev.oneie.json \
    --test_file=data/ACE05-E/test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=8 \
    --learning_rate=2e-5 \
    --weight_decay=1e-5 \
    --gradient_clip_val=5.0 \
    --accumulate_grad_batches=8 \
    --num_train_epochs=30 \
    --warmup_ratio=0.1 \
    --negative_sampling \
    --negative_ratio=1. \
    --refresh
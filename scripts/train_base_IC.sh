#!/bin/bash

python train_ic.py \
    --ckpt_name=IC-ace \
    --dataset=ACE \
    --train_file=data/ACE05-E/train.oneie.json \
    --val_file=data/ACE05-E/dev.oneie.json \
    --test_file=data/ACE05-E/test.oneie.json \
    --train_batch_size=8 \
    --eval_batch_size=16 \
    --learning_rate=2e-5 \
    --weight_decay=1e-5 \
    --gradient_clip_val=5.0 \
    --accumulate_grad_batches=8 \
    --num_train_epochs=100 \
    --warmup_ratio=0.1 \
    --negative_sampling \
    --negative_ratio=.1 \
    --refresh \
    --num_class=38 \
    --problem_type=multi_label_classification \

#!/bin/bash

srun python train_ic.py \
    --ckpt_name=IC-ace-pred \
    --load_ckpt=checkpoints/IC-ace/epoch=0.ckpt \
    --dataset=ACE \
    --eval_only \
    --train_file=data/ACE05-E/train.oneie.json \
    --val_file=data/ACE05-E/dev.oneie.json \
    --test_file=data/ACE05-E/test.oneie.json \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --accumulate_grad_batches=8 \
    --problem_type=multi_label_classification \
    --num_class=38 \
    --refresh

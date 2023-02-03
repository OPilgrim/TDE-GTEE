#!/bin/bash

python train_gtee.py \
    --model=gen \
    --ckpt_name=gen_pred \
    --dataset=ACE \
    --eval_only \
    --load_ckpt=pre-checkpoints/gen-ace/epoch=4.ckpt \
    --train_file=data/ACE05-E/train.oneie.json \
    --val_file=data/ACE05-E/dev.oneie.json \
    --test_file=data/ACE05-E/test.oneie.json \
    --eval_batch_size=32 \
    --refresh \
    --az_file=pre-checkpoints/IC-ace-pred/preprocessed/test.jsonl

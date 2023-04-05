#!/bin/bash

deepspeed \
     --include="localhost:0,1" \
     --master_port 29500 \
     --module training.trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 5 \
     --local-output-dir /home/bo_ling/dolly_training/ma_helpdesk \
     --local-data-file-path /home/bo_ling/dataset/michelangelo_so_long.hf \
     --per-device-train-batch-size 2 \
     --per-device-eval-batch-size 2 \
     --test-size 10 \
     --lr 1e-5

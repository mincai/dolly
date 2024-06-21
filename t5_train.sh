#!/bin/bash

deepspeed \
     --include="localhost:0,1" \
     --master_port 29500 \
     --module training.t5_trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 10 \
     --local-output-dir /home/bo_ling/dolly_training/eats_receipt_gcp_t5_v3 \
     --local-data-file-path /home/bo_ling/dataset/eats_receipt_gcp_v3_train.hf \
     --per-device-train-batch-size 8 \
     --per-device-eval-batch-size 8 \
     --test-size 10 \
     --lr 1e-5

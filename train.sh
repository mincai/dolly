#!/bin/bash

deepspeed \
     --include="localhost:0,1" \
     --master_port 29500 \
     --module training.trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 5 \
     --local-output-dir /home/bo_ling/dolly_training/doc_transcript_pii_data_b8 \
     --local-data-file-path /home/bo_ling/dataset/doc_transcript_pii_data.hf \
     --per-device-train-batch-size 8 \
     --per-device-eval-batch-size 8 \
     --test-size 100 \
     --lr 1e-5

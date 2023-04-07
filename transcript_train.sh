#!/bin/bash

deepspeed \
     --include="localhost:0,1" \
     --master_port 29501 \
     --module training.transcript_trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 1 \
     --local-output-dir /home/bo_ling/dolly_training/doc_transcript_pii_data_simplify_b2 \
     --local-data-file-path /home/bo_ling/dataset/doc_transcript_pii_data_simplify.hf \
     --per-device-train-batch-size 2 \
     --per-device-eval-batch-size 2 \
     --test-size 100 \
     --lr 1e-5

#!/bin/bash

deepspeed \
     --master_port 29501 \
     --module training.agent_trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 2 \
     --pretrained-model-name EleutherAI/gpt-neox-20b \
     --local-output-dir /home/bo_ling/dolly_training/modeling_data_v2 \
     --local-data-file-path /home/bo_ling/dataset/modeling_data_v2.hf \
     --per-device-train-batch-size 1 \
     --per-device-eval-batch-size 1 \
     --test-size 100 \
     --lr 1e-5

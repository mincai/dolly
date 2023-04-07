#!/bin/bash

deepspeed \
     --include="localhost:2,3" \
     --master_port 29500 \
     --module training.agent_trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 1 \
     --local-output-dir /home/bo_ling/dolly_training/agent_modeling_data_v1_expanded \
     --local-data-file-path /home/bo_ling/dataset/modeling_data_v1_expanded.hf \
     --per-device-train-batch-size 1 \
     --per-device-eval-batch-size 1 \
     --test-size 100 \
     --lr 1e-5

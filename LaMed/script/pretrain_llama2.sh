#!/bin/bash

# run "accelerate config" first!

accelerate launch --num_processes 2 LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path ./LaMed/pretrained_model/llama-2-7b-chat \
    --model_type llama2 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config ./LaMed/script/fsdp_llama2_2x24gb.json \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-pretrain-0000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 4 \
    --report_to tensorboard

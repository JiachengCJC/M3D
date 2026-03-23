#!/usr/bin/env bash
set -euo pipefail

cd /workspace/M3D

: "${MODEL_NAME_OR_PATH:=/models/phi-3-mini-128k-instruct}"
: "${PRETRAIN_VISION_MODEL:=/workspace/M3D/pretrained_ViT.bin}"
: "${OUTPUT_DIR:=/workspace/M3D/LaMed/output/LaMed-Phi3-4B-pretrain-0000}"
: "${ACCELERATE_CONFIG:=/workspace/M3D/docker/accelerate/single_gpu.yaml}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${DATALOADER_NUM_WORKERS:=1}"
: "${LEARNING_RATE:=1e-4}"

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "MODEL_NAME_OR_PATH does not exist: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAIN_VISION_MODEL}" ]]; then
  echo "PRETRAIN_VISION_MODEL does not exist: ${PRETRAIN_VISION_MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" /workspace/M3D/.cache/huggingface # Creates directories if they do not already exist.

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  LaMed/src/train/train.py \
  --version v0 \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --model_type phi3 \
  --vision_tower vit3d \
  --pretrain_vision_model "${PRETRAIN_VISION_MODEL}" \
  --tune_mm_mlp_adapter True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --eval_strategy "steps" \
  --eval_accumulation_steps 1 \
  --eval_steps 0.04 \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 1 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --gradient_checkpointing False \
  --dataloader_pin_memory True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --report_to tensorboard

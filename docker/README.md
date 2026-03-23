# Docker Pretraining (Phi-3)

This setup runs Step-1 pretraining inside Docker and mounts your local Phi-3 model:

- Host path: `/Users/jiacheng/Documents/Models/phi-3-mini-128k-instruct`
- Container path: `/models/phi-3-mini-128k-instruct`

## 1) Build image

```bash
docker compose -f docker-compose.pretrain.yml build
```

## 2) Start pretraining

```bash
docker compose -f docker-compose.pretrain.yml up
```

The launcher used is:

- `/workspace/M3D/docker/pretrain_phi3_docker.sh`

## 3) Override training knobs (optional)

Example with a larger batch size:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=2 \
docker compose -f docker-compose.pretrain.yml up
```

## Notes

- This Docker setup expects NVIDIA GPU support (`--gpus`) and `nvidia-container-toolkit`.
- By default, output is written to:
  - `/workspace/M3D/LaMed/output/LaMed-Phi3-4B-pretrain-0000`

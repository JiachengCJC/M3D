FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DS_BUILD_OPS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/M3D

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r /tmp/requirements.txt && \
    python -m pip install accelerate==0.30.1 sentencepiece protobuf==4.25.3 

COPY . /workspace/M3D

CMD ["/bin/bash"]

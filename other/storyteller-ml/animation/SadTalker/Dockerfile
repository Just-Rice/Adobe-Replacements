# syntax=docker/dockerfile:1

#FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04 # NB: Does this not work???
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# NB: SadTalker obviously requires ffmpeg
# NB: rsync to monitor time it takes for files to sync in k8s initContainer.
# NB(bt,2023-05-04): Installing lsof, htop, ripgrep, and other debugging tools
# NB: I had to remove `nvidia-driver-515` because it broke CUDA / nvidia-smi
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    htop \
    libsndfile1 \
    lsof \
    python3-pip \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    ripgrep \
    rsync \
    vim \
    wget \
    --no-install-recommends \
    && apt-get clean autoclean && apt-get autoremove -y && rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN mkdir /python_install

WORKDIR /python_install

COPY requirements.txt . 

# NB: Kubernetes will copy over the venv we create here.
RUN python3 -m venv python \
  && . python/bin/activate \
  && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
  && pip install -r requirements.txt

# Model code is kept separate from the venv
RUN mkdir /model_code

WORKDIR /model_code

COPY . .

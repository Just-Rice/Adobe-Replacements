# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04
#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# tzdata install hangs without presetting the time zone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# NB: SadTalker requires ffmpeg
# NB: rsync to monitor time it takes for files to sync in k8s initContainer.
# NB: I had to remove `nvidia-driver-515` because it broke CUDA / nvidia-smi
# NB(bt,2025-01-20): StableAnimator uses git, git-lfs
# NB(bt,2025-01-20): unzip for web installs
# NB(bt,2025-01-20): debug / utility tools: lsof, htop, ripgrep, tmux, etc.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    git-lfs \
    htop \
    libsndfile1 \
    lsof \
    python3-pip \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    ripgrep \
    rsync \
    tmux \
    unzip \
    vim \
    wget \
    --no-install-recommends \
    && apt-get clean autoclean && apt-get autoremove -y && rm -rf /var/lib/{apt,dpkg,cache,log}/

# Install rclone
RUN cd /tmp && \
    curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip && \
        unzip rclone-current-linux-amd64.zip && \
        cd rclone-*-linux-amd64 && \
        cp rclone /usr/bin/ && \
        chmod 755 /usr/bin/rclone && \
        rm -rf /tmp/rclone*

# Install entr
RUN cd /tmp && \
    wget -O entr.zip https://github.com/eradman/entr/archive/refs/tags/5.6.zip && \
        unzip entr.zip && \
        cd entr-5.6 && \
        ./configure && \
        make test && \
        make install \
        && rm -rf /tmp/entr-5.6

RUN mkdir /python_install

WORKDIR /python_install

COPY requirements.txt . 

# NB: Kubernetes will copy over the venv we create here.
# NB: pip install taken from the StableAnimator README
RUN python3 -m venv python \
  && . python/bin/activate \
  && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
  && pip install torch==2.5.1+cu124 xformers --index-url https://download.pytorch.org/whl/cu124 \
  && pip install -r requirements.txt

# Model code is kept separate from the venv
RUN mkdir /model_code

WORKDIR /model_code

COPY . .

# NB(bt,2025-01-20): Let's do this outside of the image.
#RUN git lfs install
#RUN git clone https://huggingface.co/FrancisRing/StableAnimator checkpoints

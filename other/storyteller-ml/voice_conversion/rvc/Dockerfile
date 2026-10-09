FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# NB: ffmpeg for transcoding opus -> wav
# NB(bt,2023-05-04): Installing lsof, htop, ripgrep, and other debugging tools
# NB(bt,2023-05-04): Installing rsync to monitor time it takes for files to sync
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    htop \
    libsndfile1 \
    lsof \
    nvidia-driver-515 \
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


RUN mkdir /install
WORKDIR /install

COPY requirements.txt . 


RUN python3 -m venv python \
  && . python/bin/activate \
  && pip install -r requirements.txt

WORKDIR /

COPY . /models/voice_conversion/rvc


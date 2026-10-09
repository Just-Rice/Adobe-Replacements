# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04
#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# tzdata install hangs without presetting the time zone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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
    ripgrep \
    rsync \
    tmux \
    unzip \
    vim \
    wget \
    checkinstall \
    libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev lzma liblzma-dev libbz2-dev \
    --no-install-recommends \
    && apt-get clean autoclean && apt-get autoremove -y && rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tgz && \
    tar xzf Python-3.9.21.tgz && \
    cd Python-3.9.21 && \
    ./configure --enable-optimizations && \
    make -j 8 install

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

COPY requirements*.txt .

# Kubernetes will copy over the venv we create here.
# pip install taken from a frozen virtualenv.
RUN python3 -m venv python \
  && . python/bin/activate \
  && pip install --upgrade pip wheel setuptools \
  && pip install -r requirements.frozen.txt

# Model code is kept separate from the venv
RUN mkdir /model_code

WORKDIR /model_code

COPY . .
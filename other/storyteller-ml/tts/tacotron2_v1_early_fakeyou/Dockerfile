# ==================== Python Build Base ====================

# Experimental NVidia CUDA 
#  - ubuntu 20.04
#  - nvidia-384
#  - multiple python versions (3.6, 3.8)
#  THIS IS NOT VERIFIED TO WORK YET (2021-05-30)

# NB(bt,2023-11-27): Nvida yanked their "nvidia/cuda:11.3.0-devel-ubuntu20.04" image from Docker Hub
# We've attempted to recreate our own base image replacement, but we haven't tested it in production 
# as of this writing. (This is only for TT2, which uses a very outdated set of dependencies.)
# FROM nvidia/cuda:11.3.0-devel-ubuntu20.04 
FROM ghcr.io/storytold/nvidia-cuda-11.3.0-ubuntu-20.04:508160ec7f58

# See: https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title='Storyteller ML'
LABEL org.opencontainers.image.authors='bt@brand.io, echelon@gmail.com'
LABEL org.opencontainers.image.description='Storyteller ML model monorepo: TTS, etc.'
LABEL org.opencontainers.image.documentation='https://github.com/storytold/tts-core'
LABEL org.opencontainers.image.source='https://github.com/storytold/tts-core'
LABEL org.opencontainers.image.url='https://github.com/storytold/tts-core'

# tzdata install hangs without presetting the time zone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# NB(2021-05-30): This is the image that Tacotron 2's Dockerfile specifies.
# I have not tried this, but it might be necessary.
#FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7
#ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

# NB: https://github.com/NVIDIA/nvidia-docker/issues/864#issuecomment-439848887
# NB: We do not install ffmpeg, since the version is 3.8.* series and we need 4.2.*
#
# NB(2021-05-30): We need to install `cython3` for building modern Tacotron 2
# dependencies (ie. protoc) on old Ubuntu
# RUN apt-get update \
#     && apt-get install -y \
RUN apt-get update

# NB: Here we install nvidia drivers
# NB: Avoid keyboard prompt
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
        nvidia-384 \
        --no-install-recommends

# Notes on packages:
#  * software-properties-common - needed for `apt-add-repository`
#  * pciutils - needed for `lspci | grep -i nvidia`
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        cython3 \
        ffmpeg \
        g++-7 \
        gcc-7 \
        git \
        htop \
        imagemagick \
        libc++-7-dev \
        libffi-dev \
        libgcc-7-dev \
        libmagic1 \
        libsm6 \
        libsndfile1 \
        libssl-dev \
        libxext6 \
        libxrender-dev \
        netcat \
        pciutils \
        pkg-config \
        redis-tools \
        silversearcher-ag \
        software-properties-common \
        sox \
        tmux \
        unzip \
        wget

# Wav2Lip needs Python3.6
# NB: we install python3-virtualenv as `python3.6 -m venv` won't work anymore
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y \
        python3-pip \
        python3.6 \
        python3.6-dev \
        python3.6-venv

RUN python3.6 --version

RUN python3.6 -m pip install pip

# ==================== Tacotron, Waveglow, HifiGan, etc. ====================

WORKDIR /

RUN mkdir -p /models/tts

COPY ./requirements-tacotron-python36.txt ./models/tts

WORKDIR models/tts

# NB: Not sure if we need Python3.6 for Tacotron2, but Python3.8 gave me trouble.
# NB: Setuptools fix: https://github.com/tensorflow/tensorflow/issues/34302#issuecomment-554450289
RUN python3.6 -m pip install --upgrade pip setuptools \
  && python3.6 -m pip install -r requirements-tacotron-python36.txt \
  && python3.6 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && python3.6 -m pip cache purge

# Final image
#  FROM ubuntu:xenial
#FROM pybuild-requirements as final

WORKDIR /


#COPY scripts/ ./scripts

# NB(echelon,2022-07-06): The following fix for revoked keys was copied from "storyteller-rust".
#  These dockerfiles are in such bad shape that they all need to be revisited and pared down to 
#  only the essentials.
#
# NB(echelon,2022-05-31): NVIDIA revoked some of their keys.
# A better long term fix is to remove these dependencies outright and not have ML deps in this repo.
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# https://forums.developer.nvidia.com/t/invalid-public-key-for-cuda-apt-repository/212901/11
RUN apt-key del 7fa2af80

# NB: Fix for 'E: Conflicting values set for option Signed-By regarding source'
# NB: We need to remove all sources
RUN rm $(find /etc/apt | grep cuda)

#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# ESpeak for new text pipeline
RUN apt-get install -y libespeak-ng1

RUN python3 --version

# TODO: Use venv to do this instead.
RUN pip3 install gdown && pip3 cache purge

# NB(echelon,2022-07-14): Run a newer version of pytorch which will support RTX3090.
#  Fixes bug, "NVIDIA GeForce RTX 3080 with CUDA capability sm_86 is not compatible with the current PyTorch installation."
#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Mostly for local development:
COPY ./container_start_tts_server.sh /

# TTS model code:
COPY ./ ./models/tts

COPY popular_weights.txt /model_cache_prewarm_list.txt
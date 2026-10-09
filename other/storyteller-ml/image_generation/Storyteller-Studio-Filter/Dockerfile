# Use NVIDIA CUDA with Ubuntu as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variables
ENV MY_ENV_VARIABLE=value
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /python_install
RUN mkdir /model_code

WORKDIR /python_install

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    espeak-ng \
    sudo

# rsync to copy between pods with rich logs
RUN apt-get install -y rsync

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-full python3.10-distutils

COPY requirements.txt .
# Install Python dependencies
RUN python3.10 -m venv python \
  && python/bin/pip  install --upgrade pip wheel setuptools \
  && . python/bin/activate \
  && pip install -r requirements.txt

WORKDIR /model_code

COPY . .
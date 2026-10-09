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

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install git-lfs \
    && git lfs install
# rsync to copy between pods with rich logs
RUN apt-get install -y rsync

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install -y python3.8 python3.8-venv python3.8-dev python3.8-full python3.8-distutils

COPY requirements.txt .
# Install Python dependencies
RUN python3.8 -m venv python \
  && python/bin/pip  install --upgrade pip wheel setuptools \
  && . python/bin/activate \
  && pip install -r requirements.txt

# Change working directory to StyleTTS2
WORKDIR /model_code

# Copy only the necessary files from the host to the container
COPY . .


# Clone the Git LFS repository and move the Models directory
RUN git lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS && \
    mv StyleTTS2-LibriTTS/Models .

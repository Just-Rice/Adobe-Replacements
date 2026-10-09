# ==================== BASE CONTAINER + REQUIREMENTS ====================

# NB: Make sure the CUDA version supports the driver version:
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# See: https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title='Storyteller ML : VITS'
LABEL org.opencontainers.image.description='Storyteller ML : VITS'
LABEL org.opencontainers.image.documentation='https://github.com/storytold/storyteller-ml'
LABEL org.opencontainers.image.source='https://github.com/storytold/storyteller-ml'
LABEL org.opencontainers.image.url='https://github.com/storytold/storyteller-ml'

# tzdata install hangs without presetting the time zone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Notes on packages:
#  * espeak-ng is critical for speech
#  * ffmpeg: file length detection
#  * libmagic: mimetype detection, other magic bytes operations in Python
#  * screen: not sure why included; I threw in tmux and curl for good measure
#  * unzip: not sure why included, but perhaps downloading models
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
  && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        espeak-ng \
        ffmpeg \
        libmagic-dev \
        libmagic1 \
        python3-dev \
        python3-pip \
        python3.10 \
        python3.10-venv \
        screen \
        tmux \
        unzip

RUN python3 --version

# ==================== INSTALL MORE VITS REQUIREMENTS ====================

WORKDIR /
RUN mkdir -p /vits
COPY ./requirements.txt /vits

WORKDIR /vits

RUN pip3 install --upgrade pip setuptools \
  && pip3 install -r requirements.txt \
  && pip3 install \
     transformers \
     g2p-en \
     scipy \
     text_unidecode \
     emoji \
     phonemizer \
     numpy \
     matplotlib \
     librosa==0.9.2 \
     Cython \
     Unidecode \
     tensorboard \
     gdown

# ==================== VITS CODE + FINAL BUILD ====================

COPY ./ ./

RUN cd monotonic_align \
  && mkdir monotonic_align \
  && python3 setup.py build_ext --inplace \
  && cd ..

WORKDIR /

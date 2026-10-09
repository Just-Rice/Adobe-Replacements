FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# NB: ffmpeg for transcoding opus -> wav
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    htop \
    libsndfile1 \
    nvidia-driver-515 \
    python3-pip \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    vim \
    wget \
    --no-install-recommends \
    && apt-get clean autoclean && apt-get autoremove -y && rm -rf /var/lib/{apt,dpkg,cache,log}/


RUN mkdir /install
WORKDIR /install

#RUN curl -sSL https://install.python-poetry.org | python3 -

#COPY pyproject.toml . 
#COPY poetry.lock . 
#COPY requirements.txt . 
COPY requirements-without-poetry.txt . 

#RUN pip install -r requirements-without-poetry.txt 

RUN python3 -m venv python \
  && . python/bin/activate \
  && pip install -r requirements-without-poetry.txt

WORKDIR /

COPY . /models/voice_conversion/so-vits-svc
#WORKDIR /models/voice_conversion/so-vits-svc
#RUN pip install -e .
#WORKDIR /

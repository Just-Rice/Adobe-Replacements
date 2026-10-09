
# NB(bt, 2022-12-12): The version of containerized CUDA must match (or be less than?) that of the host system running this container.
#  Juggling these dependencies, plus the underlying Ubuntu system (which controls Python versions), is tricky. 
#  Nvidia does not provide a CUDA 11.6 with Ubuntu 22.04, which mirrors my own desktop machine. Their CUDA 11.6 + Ubuntu 20.04 image has 
#  a stale Python (3.8), so we have to use the DeadSnakes PPA to install a more modern version.
#  Gross.
FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

# See: https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title='Storyteller ML: Voice Conversion'
LABEL org.opencontainers.image.authors='bt@brand.io, echelon@gmail.com'
LABEL org.opencontainers.image.description='From the Storyteller ML model monorepo: TTS, etc.'
LABEL org.opencontainers.image.documentation='https://github.com/storytold/tts-core'
LABEL org.opencontainers.image.source='https://github.com/storytold/tts-core'
LABEL org.opencontainers.image.url='https://github.com/storytold/tts-core'

# tzdata install hangs without presetting the time zone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Needed to install "future" release of Python 3.10 under Ubuntu 20.04, which only natively packages Python 3.8.
# We seem to be stuck on Ubuntu 22.04 because containerizd CUDA 11.6 is the only version we can run on my host machine,
# and Nvidia does not supply a (CUDA 11.6, Ubuntu 22.04) docker base image.
RUN apt-get update \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
        curl \
        ffmpeg \
        libmagic-dev \
        libmagic1 \
        python3-pip \
        python3.10 \
        python3.10-venv \
        unzip \
        wget \
        --no-install-recommends

# NB: Emit version for build debugging
RUN python3.10 --version

COPY requirements.txt .

# Get the correct PyTorch versions with the interactive version selector here: https://pytorch.org/
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 # For CUDA 11.6
# pip install torch torchvision torchaudio # For CUDA 11.7

# # NB: Forget about virtual environments inside the container.
# #   Also note the --extra-index-url, which is required for pytorch. 
# #   The version persisted to 'requirements.txt' chokes without it.
# RUN pip install --upgrade pip \
#     && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
#     && pip install -r requirements.txt

# NB: Install requirements in a virtual env too.
#   Also note the --extra-index-url, which is required for pytorch. 
#   The version persisted to 'requirements.txt' chokes without it.
RUN python3.10 -m venv python
RUN . python/bin/activate \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip install -r requirements.txt \
    && deactivate

# Copy remaining files
COPY . ./

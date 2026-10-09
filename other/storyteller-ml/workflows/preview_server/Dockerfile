FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG COMFY_GIT_URL=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFY_GIT_SHA=faa57430b0ff882275b1afcf6610e8e9f8a5929b

# See: https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title='Storyteller ML: Studio Runners'
LABEL org.opencontainers.image.authors='kasisnu@storyteller.ai'

# tzdata install hangs without presetting the time zone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Needed to install "future" release of Python 3.10 under Ubuntu 20.04, which only natively packages Python 3.8.
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
        python3.10-dev \
        unzip \
        htop \
        vim \
        tmux \
        rsync \
        wget \
        git \
        --no-install-recommends

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel requests argparse

RUN cd /tmp && \
    curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip && \
        unzip rclone-current-linux-amd64.zip && \
        cd rclone-*-linux-amd64 && \
        cp rclone /usr/bin/ && \
        chmod 755 /usr/bin/rclone && \
        rm -rf /tmp/rclone*

RUN mkdir -p /root/.config/pip
COPY ./pip.conf /root/.config/pip/pip.conf


# Copy files to container
USER appuser
WORKDIR /app
COPY install.py /app/install.py

# download & install comfy ui and its dependencies
RUN python3 install.py --comfy-ui-url ${COMFY_GIT_URL} --comfy-ui-commit ${COMFY_GIT_SHA}

RUN python3 -m venv venv
RUN venv/bin/python -m pip install -r ComfyUI/requirements.txt --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ && \
    venv/bin/python -m pip cache purge

COPY required_requirements.txt /app/required_requirements.txt
RUN venv/bin/python -m pip install -r required_requirements.txt --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ && \
    venv/bin/python -m pip uninstall onnxruntime onnxruntime-gpu -y && \
    venv/bin/python -m pip install onnxruntime-gpu==1.17.1  ort-nightly-gpu --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/ --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ && \
    venv/bin/python -m pip cache purge

COPY repos.json /app/repos.json
RUN python3 install.py --repos-json repos.json

COPY dependencies.json /app/dependencies.json
COPY preview_server.py /app/preview_server.py
COPY server_requirements.txt /app/server_requirements.txt
COPY initialize_comfy_extensions.py /app/initialize_comfy_extensions.py

ENV COMFY_ROOT_DIRECTORY=/app/ComfyUI
ENV V2V_WORKFLOWS_DIRECTORY=/workflow_configs
#TODO(kasisnu, 15/07/2024): remove root user after confirming that it is not needed anymore
USER root
# protobuf use python implementation (weird error without)
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
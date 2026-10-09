FROM nvidia/cuda:11.6.1-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /python_install
RUN mkdir /model_code

WORKDIR /python_install
RUN \
	echo "**** packages installation ****" \
		&& apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
		&& apt-get update && apt-get install -y \
			vim \
            build-essential \
            cmake \
            libopencv-dev \
            libjpeg-dev \
            libpng-dev \
            libglew-dev \
            libpthread-stubs0-dev \
			git \
			virtualenv \
			time \
			sudo \
			wget \
			nano \
			curl \
			software-properties-common \
			rsync \
			apt-utils \
			ffmpeg \
			unzip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install git-lfs \
    && git lfs install

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-full python3.10-distutils

# COPY requirements.txt .
# Install Python dependencies
RUN python3.10 -m venv python \
  && python/bin/pip  install --upgrade pip wheel setuptools \
  && . python/bin/activate 

WORKDIR /model_code
COPY . /model_code/MocapNET
# Fix line endings for Unix/Linux environment
RUN sed -i 's/\r$//' /model_code/MocapNET/initialize.sh
RUN sed -i 's/\r$//' /model_code/MocapNET/src/python/mnet4/setup.sh

# Execute the setup script
RUN /model_code/MocapNET/initialize.sh
RUN /model_code/MocapNET/src/python/mnet4/setup.sh --collab

WORKDIR /model_code/MocapNET/src/python/mnet4
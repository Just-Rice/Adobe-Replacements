# Use ubuntu as the base image from the hub
# then have to build this as a image.
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /python_install
RUN mkdir /model_code

WORKDIR /python_install

RUN apt-get update && apt-get install -y curl

    # Preconfigure package selections (as an example)
RUN echo "package-name package-name/some-option select option-value" | debconf-set-selections && \ 
    apt-get install -y software-properties-common

RUN apt-get install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install -y python3.8 python3.8-venv python3.8-dev python3.8-full python3.8-distutils

#RUN apt install python3.10 -y
RUN apt-get install -y git
RUN apt-get install -y ffmpeg
RUN apt-get install -y wget

# rsync to copy between pods with rich logs
RUN apt-get install -y rsync

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

# Clone the repository

COPY requirements-pinned.txt . 

RUN python3.8 -m venv python \
  && . python/bin/activate \
  && pip install -r requirements-pinned.txt

#RUN pip3 install -r requirements-pinned.txt

WORKDIR /model_code

COPY . . 

#RUN git pull
#RUN git reset --hard origin/main
#RUN git checkout 47c491a29d10c04c6f1efd01bc7b334d11cad4f2

# # Allow parameters to be passed
ENTRYPOINT ["python3", "main.py"]

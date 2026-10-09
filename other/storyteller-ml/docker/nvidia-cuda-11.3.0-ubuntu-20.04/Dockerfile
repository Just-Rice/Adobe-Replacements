# Storyteller recreation of "nvidia/cuda:11.3.0-devel-ubuntu20.04" -
#
# Nvidia pulled the "nvidia/cuda:11.3.0-devel-ubuntu20.04" family of images from Docker Hub. 
# This is a recreation of that image using our own last working images based on Nvidia's base 
# containers (`docker --history`) and what appears to be the Nvidia container source repo:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/end-of-life/11.3.0/ubuntu2004/devel/Dockerfile
#
# The following lines were all taken from `docker --history`, with the exception of `FROM`, which came from 
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/end-of-life/11.3.0/ubuntu2004/base/Dockerfile
# The "ADD" and "CMD" below are probably just noise from `docker --history`
#
# The Docker history command:
#
#   docker history --no-trunc  ghcr.io/storytold/tts-core:b5355de94e73 > output.txt
#
# The table needs to be trimmed to just the commands, and it's in reverse cronological order. The commands after 
# Nvidia's are our own, and they need to be removed. The leading "/bin/sh -c" shell commands need to be removed 
# as well.
#

#/bin/sh -c #(nop) ADD file:9d282119af0c42bc823c95b4192a3350cf2cad670622017356dd2e637762e425 in /
#/bin/sh -c #(nop)  CMD ["bash"]
FROM ubuntu:20.04 as base

ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.3 brand=tesla,driver>=418,driver<419 driver>=450
ENV NV_CUDA_CUDART_VERSION=11.3.58-1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-3

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN TARGETARCH=amd64 apt-get update && \
    apt-get install -y --no-install-recommends gnupg2 curl ca-certificates  && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION=11.3.0

RUN TARGETARCH=amd64 apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-11-3=${NV_CUDA_CUDART_VERSION}     ${NV_CUDA_COMPAT_PACKAGE}     && rm -rf /var/lib/apt/lists/*
RUN TARGETARCH=amd64 echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NGC-DL-CONTAINER-LICENSE /

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NV_CUDA_LIB_VERSION=11.3.0-1
ENV NV_NVTX_VERSION=11.3.58-1
ENV NV_LIBNPP_VERSION=11.3.3.44-1
ENV NV_LIBNPP_PACKAGE=libnpp-11-3=11.3.3.44-1
ENV NV_LIBCUSPARSE_VERSION=11.5.0.58-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-11-3
ENV NV_LIBCUBLAS_VERSION=11.4.2.10064-1
ENV NV_LIBCUBLAS_PACKAGE=libcublas-11-3=11.4.2.10064-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.9.6-1
ENV NCCL_VERSION=2.9.6-1
ENV NV_LIBNCCL_PACKAGE=libnccl2=2.9.6-1+cuda11.3

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN TARGETARCH=amd64 apt-get update && apt-get install -y --no-install-recommends     cuda-libraries-11-3=${NV_CUDA_LIB_VERSION}     ${NV_LIBNPP_PACKAGE}     cuda-nvtx-11-3=${NV_NVTX_VERSION}     libcusparse-11-3=${NV_LIBCUSPARSE_VERSION}     ${NV_LIBCUBLAS_PACKAGE}     ${NV_LIBNCCL_PACKAGE}     && rm -rf /var/lib/apt/lists/*
RUN TARGETARCH=amd64 apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY nvidia_entrypoint.sh /opt/nvidia/

ENV NVIDIA_PRODUCT_NAME=CUDA
ENV NVIDIA_CUDA_END_OF_LIFE=1

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

ENV NV_CUDA_LIB_VERSION=11.3.0-1
ENV NV_CUDA_CUDART_DEV_VERSION=11.3.58-1
ENV NV_NVML_DEV_VERSION=11.3.58-1
ENV NV_LIBCUSPARSE_DEV_VERSION=11.5.0.58-1
ENV NV_LIBNPP_DEV_VERSION=11.3.3.44-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-11-3=11.3.3.44-1
ENV NV_LIBCUBLAS_DEV_VERSION=11.4.2.10064-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-11-3
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-11-3=11.4.2.10064-1
ENV NV_NVPROF_VERSION=11.3.58-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-11-3=11.3.58-1
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.9.6-1
ENV NCCL_VERSION=2.9.6-1
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.9.6-1+cuda11.3

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN TARGETARCH=amd64 apt-get update && apt-get install -y --no-install-recommends     libtinfo5 libncursesw5     cuda-cudart-dev-11-3=${NV_CUDA_CUDART_DEV_VERSION}     cuda-command-line-tools-11-3=${NV_CUDA_LIB_VERSION}     cuda-minimal-build-11-3=${NV_CUDA_LIB_VERSION}     cuda-libraries-dev-11-3=${NV_CUDA_LIB_VERSION}     cuda-nvml-dev-11-3=${NV_NVML_DEV_VERSION}     ${NV_NVPROF_DEV_PACKAGE}     ${NV_LIBNPP_DEV_PACKAGE}     libcusparse-dev-11-3=${NV_LIBCUSPARSE_DEV_VERSION}     ${NV_LIBCUBLAS_DEV_PACKAGE}     ${NV_LIBNCCL_DEV_PACKAGE}     && rm -rf /var/lib/apt/lists/*
RUN TARGETARCH=amd64 apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

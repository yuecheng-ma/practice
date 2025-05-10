FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sudo \
    cmake \
    gdb \
    net-tools \
    zip \
    unzip \
    tar \
    gzip \
    wget \
    curl \
    vim \
    nano \
    htop \
    nvtop \
    tmux \
    screen \
    man \
    tree \
    less \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir matplotlib

WORKDIR /workspace
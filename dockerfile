# Use an official ubuntu runtime as a parent image
FROM nvidia/cuda:11.5.0-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3-dev \
    python3-tk \
    python3-pip \
    python-dipy \
    vim \
    build-essential \
    cmake \
    curl \
    git \
    libgoogle-glog-dev \
    libprotobuf-dev \
    libsm6 \
    libxext6 \
    python-tk \
    libfontconfig1 \
    libxrender1 \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    libglib2.0-0=2.50.3-2 \
    libnss3=2:3.26.2-1.1+deb9u1 \
    libgconf-2-4=3.2.6-4+b1 

RUN  apt-get update \
    && apt-get install -y wget gpg-agent \
    && rm -rf /var/lib/apt/lists/*

# Install some python packages
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir future hypothesis

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
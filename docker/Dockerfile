FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

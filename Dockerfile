# Dockerfile for FastAPI application with Pipenv and CUDA support
# Base image with CUDA 12.1
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, git and other build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN ldconfig
# Make python3.10 the default python and pip3 point to python3.10's pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    python3 -m pip install --upgrade pip

# Install pipenv
RUN pip3 install pipenv

# Set environment variables for HuggingFace tokenizers and Python
ENV TOKENIZERS_PARALLELISM="false"
ENV PYTHONUNBUFFERED=1
# REMOVE the following line if you want GPU access:
# ENV CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# Install PyTorch with CUDA 12.1 support
# Verify the correct command from https://pytorch.org/get-started/locally/
# This command is for PyTorch compatible with CUDA 12.1
# Install CPU-only PyTorch to ensure compatibility and avoid CUDA-related import errors
# when a GPU environment is not available, not correctly configured, or not intended.
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy dependency definitions
COPY Pipfile Pipfile.lock /app/

# Install application dependencies using pipenv
# Pipenv should recognize the pre-installed PyTorch if versions are compatible
RUN pipenv install --system --deploy

# Copy application source code
COPY ./src /app/src
COPY ./.env.example /app/.env.example
# Ensure your .dockerignore is correctly configured

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
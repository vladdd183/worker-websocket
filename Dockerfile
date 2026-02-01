# Base image with CUDA 12.1 and cuDNN for optimal performance
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (required for PyTorch)
RUN pip install --no-cache-dir numpy

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2 (requires CUDA development tools)
# Skip if build fails - it's optional optimization
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Flash Attention not installed - will use standard attention"

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Model will be downloaded on first run (cold start ~60-90 sec)
# Pre-downloading during build requires GPU which is not available at build time

# Copy application files
COPY asr_engine.py /app/
COPY websocket_handler.py /app/
COPY audio_utils.py /app/
COPY rp_handler.py /app/

# Expose ports
# 8765 - WebSocket for real-time streaming
# 8000 - HTTP for RunPod handler (if needed)
EXPOSE 8765 8000

# Set environment variables for optimization
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false

# Start the handler
CMD ["python", "-u", "rp_handler.py"]

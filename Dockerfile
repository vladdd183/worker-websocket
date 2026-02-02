# Используем официальный NVIDIA NeMo контейнер как базу
# Это гарантирует совместимость всех зависимостей (PyTorch, CUDA, NeMo)
FROM nvcr.io/nvidia/nemo:24.09

# NeMo 24.09 содержит:
# - PyTorch 2.4.0a0
# - NeMo 2.0.0
# - CUDA 12.x
# - Все зависимости уже установлены и совместимы!

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional system dependencies (если не установлены)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install additional Python dependencies
# NeMo уже установлен в базовом образе, добавляем только недостающее
# websockets==10.4 для совместимости с handler(websocket, path) сигнатурой
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    websockets==10.4 \
    aiohttp>=3.9.0 \
    pydub>=0.25.0

# Copy application files
COPY asr_engine.py /app/
COPY websocket_handler.py /app/
COPY audio_utils.py /app/
COPY rp_handler.py /app/

# Expose ports
# 8765 - WebSocket for real-time streaming
EXPOSE 8765 8000

# Set environment variables for optimization
ENV CUDA_MODULE_LOADING=LAZY
ENV TOKENIZERS_PARALLELISM=false

# ВАЖНО: Отключаем CUDA Graphs для streaming (динамические размеры входов)
ENV NEMO_DISABLE_CUDAGRAPHS=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Pre-download model during build (optional, speeds up cold start)
# Uncomment if you want model baked into image (~2.5GB larger image)
# RUN python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"

# Start the handler
CMD ["python", "-u", "rp_handler.py"]

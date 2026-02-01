# Parakeet ASR RunPod Service

Высокопроизводительный сервис распознавания речи на базе **NVIDIA parakeet-tdt-0.6b-v3** для RunPod.

## Возможности

- **25 языков** — русский, английский, немецкий и другие европейские языки
- **Real-time WebSocket** — streaming транскрипция с низкой latency
- **Batch HTTP** — обработка длинных аудио файлов
- **Автоматические timestamps** — на уровне слов и сегментов
- **Оптимизации** — Flash Attention 2, BF16, torch.compile

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                   RunPod Worker                      │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │  WebSocket  │    │     HTTP Endpoint        │   │
│  │  Real-time  │    │     Batch Processing     │   │
│  │  :8765      │    │     (RunPod API)         │   │
│  └──────┬──────┘    └───────────┬──────────────┘   │
│         │                       │                   │
│         └───────────┬───────────┘                   │
│                     ▼                               │
│  ┌─────────────────────────────────────────────┐   │
│  │          ASR Engine (Shared Model)          │   │
│  │  - parakeet-tdt-0.6b-v3 (600M params)       │   │
│  │  - Flash Attention 2 + BF16                 │   │
│  │  - torch.compile + CUDA Graphs              │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Рекомендуемые GPU

| GPU | VRAM | Цена/сек | Рекомендация |
|-----|------|----------|--------------|
| **AMPERE 24GB** | 24GB | $0.00019 | **Оптимальный выбор** — A10/RTX 3090 |
| ADA 24GB PRO | 24GB | $0.00031 | Максимальная скорость — L4/RTX 4090 |
| AMPERE 48GB | 48GB | $0.00034 | Для очень длинных аудио — A6000/A40 |

## Быстрый старт

### 1. Сборка Docker образа

```bash
docker build -t parakeet-asr .
```

### 2. Локальный тест

```bash
docker run --gpus all -p 8765:8765 parakeet-asr
```

### 3. Деплой на RunPod

1. Загрузите образ на Docker Hub
2. Создайте Serverless Endpoint на RunPod
3. Настройте TCP порт 8765 для WebSocket

## Использование

### WebSocket (Real-time)

```python
import asyncio
from client import ASRWebSocketClient, stream_file_to_websocket

# Стриминг файла
asyncio.run(stream_file_to_websocket(
    websocket_url="ws://YOUR_IP:PORT",
    audio_file="audio.wav",
    chunk_duration=2.0
))
```

**Протокол WebSocket:**

```json
// 1. Подключение -> получаем welcome
{"type": "welcome", "message": "Подключено к Parakeet ASR"}

// 2. Начало стриминга
{"action": "start", "sample_rate": 16000, "chunk_duration": 2.0}

// 3. Отправка аудио (бинарные данные PCM16 LE)

// 4. Получение транскрипции
{"type": "transcription", "text": "привет мир", "is_final": false}

// 5. Остановка
{"action": "stop"}
```

### HTTP Batch (RunPod API)

```python
import asyncio
from client import batch_transcribe_runpod

asyncio.run(batch_transcribe_runpod(
    endpoint_url="https://api.runpod.ai/v2/YOUR_ENDPOINT",
    api_key="YOUR_API_KEY",
    audio_file="long_audio.mp3",
    timestamps=True
))
```

**Примеры запросов:**

```bash
# Транскрипция из URL
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.wav",
      "timestamps": true
    }
  }'

# Транскрипция из base64
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_base64": "UklGR...",
      "format": "wav",
      "timestamps": true
    }
  }'

# Запуск WebSocket режима
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "websocket"}}'
```

## Параметры

### Batch транскрипция

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `audio_base64` | string | - | Base64 аудио |
| `audio_url` | string | - | URL аудио файла |
| `timestamps` | bool | true | Включить timestamps |
| `use_local_attention` | bool | false | Для аудио > 24 мин |
| `chunk_long_audio` | bool | false | Разбить на чанки |
| `chunk_duration` | float | 30.0 | Длина чанка (сек) |

### WebSocket конфигурация

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `sample_rate` | int | 16000 | Частота дискретизации |
| `chunk_duration` | float | 2.0 | Длина чанка для обработки |
| `timestamps` | bool | true | Включить timestamps |

## Поддерживаемые языки

Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), 
**English (en)**, Estonian (et), Finnish (fi), French (fr), German (de), 
Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt), 
Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), **Russian (ru)**, 
Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), Ukrainian (uk)

## Поддерживаемые форматы аудио

- WAV (рекомендуется)
- FLAC
- MP3
- OGG
- WebM

## Оптимизации

1. **Flash Attention 2** — ускорение attention слоёв
2. **BF16 Precision** — 2x экономия памяти на Ampere+
3. **torch.compile** — JIT-компиляция (20-30% ускорение)
4. **Local Attention** — для аудио > 24 минут
5. **CUDA Graphs** — устранение kernel launch overhead
6. **Warmup** — стабильная latency с первого запроса

## Структура проекта

```
/
├── Dockerfile           # CUDA 12.1 + NeMo + Flash Attention
├── requirements.txt     # Python зависимости
├── rp_handler.py       # RunPod handler (dual-mode)
├── asr_engine.py       # ASR движок с оптимизациями
├── websocket_handler.py # WebSocket сервер
├── audio_utils.py      # Утилиты для аудио
├── client.py           # Тестовый клиент
└── README.md           # Документация
```

## Производительность

Модель parakeet-tdt-0.6b-v3 показывает отличные результаты:

- **Русский** (FLEURS): 5.51% WER
- **Английский** (LibriSpeech clean): 1.93% WER
- **Украинский** (FLEURS): 6.79% WER

Скорость обработки на A10 (24GB):
- ~10-15x быстрее реального времени для коротких аудио
- ~5-8x для длинных аудио с local attention

## Лицензия

- Код: MIT
- Модель: CC-BY-4.0 (NVIDIA)

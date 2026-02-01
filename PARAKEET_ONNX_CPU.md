# Parakeet TDT 0.6B V3 — ONNX для CPU

## Обзор

Для серверов **без GPU** лучшее решение — `onnx-asr` с ONNX-версией модели Parakeet.

- **Качество идентичное** оригинальной NeMo модели (WER 6.75% vs 6.76%)
- Работает на любом CPU (x86, ARM)
- Поддержка batch processing
- Минимальные зависимости (не нужен PyTorch, NeMo, FFmpeg)

## Бенчмарки (RTFx = во сколько раз быстрее реального времени)

| Платформа | RTFx | 1 мин аудио за | 1 час аудио за |
|-----------|------|----------------|----------------|
| CPU (i7-7700HQ, 4 ядра) | 9.7x | ~6 сек | ~6 мин |
| CPU (32 ядра, оценка) | ~30-50x | ~1-2 сек | ~1-2 мин |
| CPU INT8 (32 ядра, оценка) | ~50-80x | <1 сек | ~45-70 сек |
| ARM (Cortex-A53) | N/A | — | — |

### Сравнение с GPU

| Платформа | RTFx |
|-----------|------|
| CPU 32 ядра | ~30-50x |
| GPU T4 CUDA | 77x |
| GPU T4 TensorRT FP16 | 227x |
| GPU RTX 4090 (оценка) | 400-600x |

## Установка

```bash
pip install onnx-asr[cpu,hub]
```

## Использование

### Базовое распознавание

```python
import onnx_asr

# Загрузка модели (первый раз скачивает ~2.5GB)
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

# Распознавание файла
result = model.recognize("audio.wav")
print(result)
```

### INT8 версия (быстрее, чуть меньше качество)

```python
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")
```

### Batch processing (параллельная обработка)

```python
# Параллельная обработка нескольких файлов
results = model.recognize([
    "file1.wav",
    "file2.wav", 
    "file3.wav",
    "file4.wav"
])

for i, result in enumerate(results):
    print(f"File {i+1}: {result}")
```

### С timestamps (временные метки)

```python
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3").with_timestamps()
result = model.recognize("audio.wav")
# Возвращает токены с временными метками
```

### Длинные аудио (VAD сегментация)

```python
# Для аудио длиннее 30 сек используй VAD
vad = onnx_asr.load_vad("silero")
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3").with_vad(vad)

for segment in model.recognize("long_audio.wav"):
    print(segment)
```

### CLI (командная строка)

```bash
onnx-asr nemo-parakeet-tdt-0.6b-v3 audio.wav
```

## Поддерживаемые форматы

- WAV: PCM_U8, PCM_16, PCM_24, PCM_32
- Для других форматов используй `soundfile`:

```python
import soundfile as sf

waveform, sample_rate = sf.read("audio.mp3", dtype="float32")
result = model.recognize(waveform, sample_rate=sample_rate)
```

## Ограничения

| Ограничение | Решение |
|-------------|---------|
| Макс. длина аудио 20-30 сек | Используй VAD (`with_vad`) |
| Не подходит для real-time streaming | Используй GPU версию |
| Нет Flash Attention | Компенсируется ONNX оптимизациями |

## Качество (WER на English Voxpopuli)

| Модель | WER |
|--------|-----|
| NeMo original | 6.76% |
| ONNX (onnx-asr) | 6.75% |
| **Разница** | **~0%** |

## Когда использовать CPU версию

✅ **Подходит:**
- Batch обработка больших файлов
- Сервер без GPU
- Асинхронные задачи
- Экономия на GPU ресурсах

❌ **Не подходит:**
- Real-time streaming с микрофона
- Минимальная задержка критична
- Обработка сотен часов в день

## Альтернативные модели для CPU

| Модель | RTFx CPU | Качество | Язык |
|--------|----------|----------|------|
| `nemo-parakeet-tdt-0.6b-v3` | 9.7x | Лучшее | Multilingual |
| `gigaam-v3-ctc` | 14.5x | Хорошее | Русский |
| `nemo-fastconformer-ru-ctc` | 45.8x | Хорошее | Русский |
| `t-tech/t-one` | 11.7x | Хорошее | Русский |

## Ссылки

- [onnx-asr GitHub](https://github.com/istupakov/onnx-asr)
- [Parakeet ONNX модель](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- [INT8 версия](https://huggingface.co/nasedkinpv/parakeet-tdt-0.6b-v3-onnx-int8)
- [Оригинальная модель NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

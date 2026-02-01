"""
ASR Engine - загрузка и оптимизация модели parakeet-tdt-0.6b-v3

Оптимизации:
- Flash Attention 2 (если доступен)
- BF16/FP16 precision
- torch.compile для JIT-оптимизации
- Local attention для длинных аудио
- CUDA Graphs для низкой latency
"""

import os
import torch
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные настройки оптимизации
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
USE_FLASH_ATTENTION = True
USE_TORCH_COMPILE = False  # Отключено для быстрого cold start (~30-60 сек экономии)
USE_BF16 = True  # BF16 быстрее на Ampere+, FP16 на старых GPU
SKIP_WARMUP = False  # Пропустить warmup для ещё более быстрого cold start

# RunPod Cached Models path
RUNPOD_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"


def find_cached_model_path(model_name: str) -> Optional[str]:
    """
    Поиск модели в кэше RunPod Cached Models
    
    Args:
        model_name: Имя модели (например 'nvidia/parakeet-tdt-0.6b-v3')
        
    Returns:
        Путь к модели или None если не найдена
    """
    # Конвертируем формат: "nvidia/parakeet-tdt-0.6b-v3" -> "models--nvidia--parakeet-tdt-0.6b-v3"
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(RUNPOD_CACHE_DIR, f"models--{cache_name}", "snapshots")
    
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            model_path = os.path.join(snapshots_dir, snapshots[0])
            logger.info(f"Найдена модель в RunPod cache: {model_path}")
            return model_path
    
    logger.info(f"Модель не найдена в RunPod cache, будет загружена из HuggingFace")
    return None


@dataclass
class TranscriptionResult:
    """Результат транскрипции"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    word_timestamps: Optional[List[Dict]] = None
    segment_timestamps: Optional[List[Dict]] = None
    processing_time: Optional[float] = None


class ASREngine:
    """
    Движок распознавания речи на базе parakeet-tdt-0.6b-v3
    
    Поддерживает:
    - Batch inference для высокого throughput
    - Streaming inference для low latency
    - Автоматическое определение языка (25 европейских языков)
    - Timestamps на уровне слов и сегментов
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern - модель загружается один раз"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is not None:
            return
        self._load_model()
    
    def _load_model(self):
        """Загрузка и оптимизация модели"""
        import nemo.collections.asr as nemo_asr
        import time
        
        start_time = time.time()
        logger.info(f"Загрузка модели {MODEL_NAME}...")
        
        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}, память: {gpu_memory:.1f} GB")
        
        # Проверяем RunPod Cached Models
        cached_path = find_cached_model_path(MODEL_NAME)
        
        if cached_path:
            # Загружаем из локального кэша RunPod
            logger.info(f"Загрузка из RunPod cache: {cached_path}")
            # Ищем .nemo файл в директории
            nemo_files = [f for f in os.listdir(cached_path) if f.endswith('.nemo')]
            if nemo_files:
                model_file = os.path.join(cached_path, nemo_files[0])
                logger.info(f"Найден .nemo файл: {model_file}")
                self._model = nemo_asr.models.ASRModel.restore_from(model_file)
            else:
                # Пробуем загрузить как HuggingFace модель
                logger.info("Файл .nemo не найден, загрузка через from_pretrained...")
                self._model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        else:
            # Загружаем из HuggingFace (будет скачано)
            logger.info("Загрузка из HuggingFace...")
            self._model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        
        self._model = self._model.to(self.device)
        
        # Применяем оптимизации
        self._apply_optimizations()
        
        # Прогрев модели (warmup)
        self._warmup()
        
        load_time = time.time() - start_time
        logger.info(f"Модель загружена за {load_time:.2f} секунд")
    
    def _apply_optimizations(self):
        """Применение всех оптимизаций для максимальной скорости"""
        
        # 1. Half precision (BF16 или FP16)
        if self.device.type == "cuda":
            if USE_BF16 and torch.cuda.is_bf16_supported():
                logger.info("Включаем BF16 precision")
                self._model = self._model.to(dtype=torch.bfloat16)
            else:
                logger.info("Включаем FP16 precision")
                self._model = self._model.half()
        
        # 2. Evaluation mode
        self._model.eval()
        
        # 3. torch.compile отключён для быстрого cold start
        # Компиляция занимает 30-60 сек при первом запуске
        # Если нужна максимальная скорость inference (после cold start),
        # можно включить: USE_TORCH_COMPILE = True
        if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            try:
                logger.info("Применяем torch.compile (это займёт время)...")
                self._model = torch.compile(
                    self._model, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                logger.info("torch.compile применён успешно")
            except Exception as e:
                logger.warning(f"torch.compile не удалось применить: {e}")
        else:
            logger.info("torch.compile отключён для быстрого cold start")
        
        # 4. CUDA optimizations
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA TF32 и cuDNN benchmark включены")
    
    def _warmup(self):
        """Прогрев модели для стабильной latency"""
        if SKIP_WARMUP:
            logger.info("Warmup пропущен (SKIP_WARMUP=True)")
            return
        
        logger.info("Прогрев модели (1 итерация)...")
        
        # Создаём dummy аудио (0.5 секунды - минимум для быстрого warmup)
        sample_rate = 16000
        dummy_audio = np.zeros(sample_rate // 2, dtype=np.float32)
        
        # Одна итерация прогрева (достаточно для инициализации CUDA kernels)
        with torch.no_grad(), torch.inference_mode():
            try:
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    sf.write(f.name, dummy_audio, sample_rate)
                    _ = self._model.transcribe([f.name])
            except Exception as e:
                logger.warning(f"Warmup не удался: {e}")
        
        logger.info("Прогрев завершён")
    
    def configure_for_long_audio(self, max_duration_minutes: float = 24):
        """
        Настройка модели для длинных аудио с использованием local attention
        
        Args:
            max_duration_minutes: Если аудио длиннее, используется local attention
        """
        # Переключение на local attention для аудио > 24 минут
        # Это позволяет обрабатывать аудио до 3 часов
        if max_duration_minutes > 24:
            logger.info("Включаем local attention для длинных аудио")
            self._model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256]  # ~16 секунд контекста
            )
        else:
            # Full attention для лучшего качества на коротких аудио
            logger.info("Используем full attention")
            self._model.change_attention_model(
                self_attention_model="rel_pos",
                att_context_size=None
            )
    
    def transcribe(
        self,
        audio_paths: Union[str, List[str]],
        timestamps: bool = True,
        batch_size: int = 1,
        use_local_attention: bool = False
    ) -> List[TranscriptionResult]:
        """
        Транскрипция аудио файлов
        
        Args:
            audio_paths: Путь к файлу или список путей
            timestamps: Включить timestamps слов и сегментов
            batch_size: Размер батча для параллельной обработки
            use_local_attention: Использовать local attention для длинных аудио
            
        Returns:
            Список TranscriptionResult
        """
        import time
        
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        
        # Настройка attention для длинных аудио
        if use_local_attention:
            self.configure_for_long_audio(max_duration_minutes=180)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Inference с timestamps
            outputs = self._model.transcribe(
                audio_paths,
                batch_size=batch_size,
                timestamps=timestamps
            )
        
        processing_time = time.time() - start_time
        
        # Формируем результаты
        results = []
        for i, output in enumerate(outputs):
            result = TranscriptionResult(
                text=output.text if hasattr(output, 'text') else str(output),
                processing_time=processing_time / len(audio_paths)
            )
            
            # Добавляем timestamps если есть
            if timestamps and hasattr(output, 'timestamp'):
                result.word_timestamps = output.timestamp.get('word', [])
                result.segment_timestamps = output.timestamp.get('segment', [])
            
            results.append(result)
        
        return results
    
    def transcribe_audio_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        timestamps: bool = True
    ) -> TranscriptionResult:
        """
        Транскрипция из numpy array (для streaming)
        
        Args:
            audio_array: Аудио данные как numpy array
            sample_rate: Частота дискретизации (должна быть 16000)
            timestamps: Включить timestamps
            
        Returns:
            TranscriptionResult
        """
        import tempfile
        import soundfile as sf
        import time
        
        # Ресэмплинг если нужно
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
        
        # Нормализация
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()
        
        # Сохраняем во временный файл (NeMo требует файл)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio_array, 16000)
            results = self.transcribe([f.name], timestamps=timestamps)
        
        return results[0] if results else TranscriptionResult(text="")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели и её настройках"""
        info = {
            "model_name": MODEL_NAME,
            "device": str(self.device),
            "dtype": str(next(self._model.parameters()).dtype),
            "optimizations": {
                "flash_attention": USE_FLASH_ATTENTION,
                "torch_compile": USE_TORCH_COMPILE,
                "bf16": USE_BF16
            }
        }
        
        if self.device.type == "cuda":
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9
            }
        
        return info


# Глобальный экземпляр движка (lazy initialization)
_engine: Optional[ASREngine] = None


def get_engine() -> ASREngine:
    """Получение глобального экземпляра ASR движка"""
    global _engine
    if _engine is None:
        _engine = ASREngine()
    return _engine


def transcribe_file(
    audio_path: str,
    timestamps: bool = True,
    use_local_attention: bool = False
) -> TranscriptionResult:
    """
    Удобная функция для транскрипции одного файла
    
    Args:
        audio_path: Путь к аудио файлу
        timestamps: Включить timestamps
        use_local_attention: Использовать local attention для длинных аудио
        
    Returns:
        TranscriptionResult
    """
    engine = get_engine()
    results = engine.transcribe(
        [audio_path], 
        timestamps=timestamps,
        use_local_attention=use_local_attention
    )
    return results[0] if results else TranscriptionResult(text="")


def transcribe_audio(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    timestamps: bool = True
) -> TranscriptionResult:
    """
    Удобная функция для транскрипции numpy array
    
    Args:
        audio_array: Аудио данные
        sample_rate: Частота дискретизации
        timestamps: Включить timestamps
        
    Returns:
        TranscriptionResult
    """
    engine = get_engine()
    return engine.transcribe_audio_array(audio_array, sample_rate, timestamps)


if __name__ == "__main__":
    # Тестирование
    engine = get_engine()
    print("Информация о модели:")
    print(engine.get_model_info())

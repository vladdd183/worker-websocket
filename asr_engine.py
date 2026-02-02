"""
ASR Engine - загрузка и оптимизация модели parakeet-tdt-0.6b-v3

Совместимость: NeMo 2.0.0 (контейнер nvcr.io/nvidia/nemo:24.09)

Оптимизации:
- BF16/FP16 precision
- Local attention для длинных аудио
- CUDA optimizations (без CUDA Graphs для streaming)
"""

import os

# ВАЖНО: Отключаем CUDA Graphs ДО импорта torch/nemo
# CUDA Graphs не совместимы с динамическими размерами входных данных (streaming)
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
os.environ["NEMO_DISABLE_CUDAGRAPHS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Отключаем CUDA Graphs в PyTorch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 1
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные настройки
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
USE_BF16 = True  # BF16 быстрее на Ampere+
SKIP_WARMUP = False

# RunPod Cached Models path
RUNPOD_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"


def find_cached_model_path(model_name: str) -> Optional[str]:
    """Поиск модели в кэше RunPod"""
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
    Совместим с NeMo 2.0.0
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
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
            logger.info(f"Загрузка из RunPod cache: {cached_path}")
            nemo_files = [f for f in os.listdir(cached_path) if f.endswith('.nemo')]
            if nemo_files:
                model_file = os.path.join(cached_path, nemo_files[0])
                logger.info(f"Найден .nemo файл: {model_file}")
                self._model = nemo_asr.models.ASRModel.restore_from(model_file)
            else:
                logger.info("Файл .nemo не найден, загрузка через from_pretrained...")
                self._model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        else:
            logger.info("Загрузка из HuggingFace...")
            self._model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        
        self._model = self._model.to(self.device)
        
        # Применяем оптимизации
        self._apply_optimizations()
        
        # Прогрев модели
        self._warmup()
        
        load_time = time.time() - start_time
        logger.info(f"Модель загружена за {load_time:.2f} секунд")
    
    def _apply_optimizations(self):
        """Применение оптимизаций"""
        
        # ВАЖНО: Отключаем CUDA Graphs в декодере TDT
        # Это решает ошибку "CUDAGraph::replay without capture"
        self._disable_cuda_graph_decoder()
        
        # Half precision
        if self.device.type == "cuda":
            if USE_BF16 and torch.cuda.is_bf16_supported():
                logger.info("Включаем BF16 precision")
                self._model = self._model.to(dtype=torch.bfloat16)
            else:
                logger.info("Включаем FP16 precision")
                self._model = self._model.half()
        
        # Evaluation mode
        self._model.eval()
        
        # CUDA optimizations для streaming
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # ВАЖНО: benchmark=False для streaming с динамическими размерами
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            logger.info("CUDA TF32 включён, cuDNN benchmark отключён (streaming mode)")
    
    def _disable_cuda_graph_decoder(self):
        """Отключение CUDA Graphs в TDT декодере"""
        try:
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
            
            # Создаём конфиг без CUDA Graphs
            decoding_cfg = RNNTDecodingConfig(
                strategy="greedy_batch",
                model_type="tdt",
                fused_batch_size=-1
            )
            decoding_cfg.greedy.loop_labels = True
            decoding_cfg.greedy.use_cuda_graph_decoder = False  # Отключаем CUDA Graphs!
            
            # Применяем конфиг
            self._model.change_decoding_strategy(decoding_cfg)
            logger.info("CUDA Graph decoder отключён для TDT модели")
            
        except Exception as e:
            logger.warning(f"Не удалось отключить CUDA Graph decoder: {e}")
            # Пробуем альтернативный способ
            try:
                if hasattr(self._model, 'decoding') and hasattr(self._model.decoding, 'decoding'):
                    if hasattr(self._model.decoding.decoding, 'use_cuda_graph_decoder'):
                        self._model.decoding.decoding.use_cuda_graph_decoder = False
                        logger.info("CUDA Graph decoder отключён (альтернативный способ)")
            except Exception as e2:
                logger.warning(f"Альтернативный способ тоже не сработал: {e2}")
    
    def _warmup(self):
        """Прогрев модели"""
        if SKIP_WARMUP:
            logger.info("Warmup пропущен")
            return
        
        logger.info("Прогрев модели...")
        
        sample_rate = 16000
        dummy_audio = np.zeros(sample_rate // 2, dtype=np.float32)
        
        with torch.no_grad(), torch.inference_mode():
            try:
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    sf.write(f.name, dummy_audio, sample_rate)
                    _ = self._model.transcribe([f.name], batch_size=1)
            except Exception as e:
                logger.warning(f"Warmup не удался: {e}")
        
        logger.info("Прогрев завершён")
    
    def configure_for_long_audio(self, use_local: bool = True):
        """
        Настройка модели для длинных аудио
        
        Args:
            use_local: True для local attention (длинные аудио), False для full attention
        """
        try:
            if use_local:
                logger.info("Включаем local attention для длинных аудио")
                self._model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[256, 256]
                )
            else:
                logger.info("Используем full attention")
                self._model.change_attention_model(
                    self_attention_model="rel_pos",
                    att_context_size=None
                )
        except Exception as e:
            logger.warning(f"Не удалось изменить attention model: {e}")
    
    def transcribe(
        self,
        audio_paths: Union[str, List[str]],
        batch_size: int = 1,
        use_local_attention: bool = False
    ) -> List[TranscriptionResult]:
        """
        Транскрипция аудио файлов
        
        Args:
            audio_paths: Путь к файлу или список путей
            batch_size: Размер батча
            use_local_attention: Использовать local attention для длинных аудио
            
        Returns:
            Список TranscriptionResult
        """
        import time
        
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        
        # Настройка attention
        if use_local_attention:
            self.configure_for_long_audio(use_local=True)
        
        start_time = time.time()
        
        # Очищаем CUDA кэш перед транскрипцией (избегаем CUDA Graph проблем)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # NeMo 2.0.0 transcribe API
            outputs = self._model.transcribe(
                audio_paths,
                batch_size=batch_size,
                return_hypotheses=True,  # Для получения timestamps
                verbose=False
            )
        
        processing_time = time.time() - start_time
        
        # Формируем результаты
        results = []
        for i, output in enumerate(outputs):
            # output может быть строкой или объектом Hypothesis
            if hasattr(output, 'text'):
                text = output.text
            elif isinstance(output, str):
                text = output
            else:
                text = str(output)
            
            result = TranscriptionResult(
                text=text,
                processing_time=processing_time / len(audio_paths)
            )
            
            # Пробуем получить timestamps (если доступны)
            if hasattr(output, 'timestep'):
                result.word_timestamps = getattr(output, 'timestep', {}).get('word', [])
                result.segment_timestamps = getattr(output, 'timestep', {}).get('segment', [])
            elif hasattr(output, 'timestamp'):
                result.word_timestamps = getattr(output, 'timestamp', {}).get('word', [])
                result.segment_timestamps = getattr(output, 'timestamp', {}).get('segment', [])
            
            results.append(result)
        
        return results
    
    def transcribe_audio_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Транскрипция из numpy array (для streaming)
        
        Args:
            audio_array: Аудио данные
            sample_rate: Частота дискретизации
            
        Returns:
            TranscriptionResult
        """
        import tempfile
        import soundfile as sf
        
        # Ресэмплинг если нужно
        if sample_rate != 16000:
            try:
                import librosa
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            except ImportError:
                logger.warning("librosa не установлен, ресэмплинг пропущен")
        
        # Нормализация
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio_array, 16000)
            results = self.transcribe([f.name])
        
        return results[0] if results else TranscriptionResult(text="")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели"""
        info = {
            "model_name": MODEL_NAME,
            "device": str(self.device),
            "dtype": str(next(self._model.parameters()).dtype),
            "nemo_version": "2.0.0"
        }
        
        if self.device.type == "cuda":
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9
            }
        
        return info


# Глобальный экземпляр
_engine: Optional[ASREngine] = None


def get_engine() -> ASREngine:
    """Получение глобального экземпляра ASR движка"""
    global _engine
    if _engine is None:
        _engine = ASREngine()
    return _engine


def transcribe_file(
    audio_path: str,
    use_local_attention: bool = False
) -> TranscriptionResult:
    """Транскрипция файла"""
    engine = get_engine()
    results = engine.transcribe(
        [audio_path], 
        use_local_attention=use_local_attention
    )
    return results[0] if results else TranscriptionResult(text="")


def transcribe_audio(
    audio_array: np.ndarray,
    sample_rate: int = 16000
) -> TranscriptionResult:
    """Транскрипция numpy array"""
    engine = get_engine()
    return engine.transcribe_audio_array(audio_array, sample_rate)


if __name__ == "__main__":
    engine = get_engine()
    print("Информация о модели:")
    print(engine.get_model_info())

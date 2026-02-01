"""
Audio Utilities - работа с аудио форматами и буферизация

Поддерживает:
- Конвертация различных форматов в WAV 16kHz mono
- Буферизация аудио потока для streaming
- Декодирование base64 аудио
- Загрузка аудио из URL
"""

import io
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def load_audio_file(
    file_path: str,
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудио файла и конвертация в нужный формат
    
    Args:
        file_path: Путь к файлу
        target_sr: Целевая частота дискретизации
        
    Returns:
        Tuple[audio_array, sample_rate]
    """
    import librosa
    
    # Загружаем с конвертацией в mono и ресэмплингом
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    return audio.astype(np.float32), sr


def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE,
    format_hint: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудио из байтов
    
    Args:
        audio_bytes: Аудио данные в байтах
        target_sr: Целевая частота дискретизации
        format_hint: Подсказка формата (wav, mp3, ogg, etc.)
        
    Returns:
        Tuple[audio_array, sample_rate]
    """
    import soundfile as sf
    import librosa
    
    # Пробуем разные методы загрузки
    try:
        # Сначала пробуем soundfile (быстрее для wav/flac)
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        try:
            # Пробуем pydub для других форматов
            from pydub import AudioSegment
            
            if format_hint:
                segment = AudioSegment.from_file(
                    io.BytesIO(audio_bytes), 
                    format=format_hint
                )
            else:
                segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Конвертируем в numpy array
            audio = np.array(segment.get_array_of_samples(), dtype=np.float32)
            audio /= 32768.0  # Нормализация для int16
            sr = segment.frame_rate
            
            # Если стерео, конвертируем в mono
            if segment.channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
                
        except Exception as e:
            raise ValueError(f"Не удалось декодировать аудио: {e}")
    
    # Конвертируем в mono если нужно
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Ресэмплинг если нужно
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio.astype(np.float32), sr


def load_audio_from_base64(
    base64_string: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    format_hint: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудио из base64 строки
    
    Args:
        base64_string: Base64 закодированные аудио данные
        target_sr: Целевая частота дискретизации
        format_hint: Подсказка формата
        
    Returns:
        Tuple[audio_array, sample_rate]
    """
    # Декодируем base64
    audio_bytes = base64.b64decode(base64_string)
    return load_audio_from_bytes(audio_bytes, target_sr, format_hint)


async def load_audio_from_url(
    url: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    timeout: int = 60
) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудио из URL
    
    Args:
        url: URL аудио файла
        target_sr: Целевая частота дискретизации
        timeout: Таймаут загрузки в секундах
        
    Returns:
        Tuple[audio_array, sample_rate]
    """
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                raise ValueError(f"Ошибка загрузки: HTTP {response.status}")
            
            audio_bytes = await response.read()
            
            # Определяем формат по content-type или расширению
            content_type = response.headers.get('content-type', '')
            format_hint = None
            
            if 'mp3' in content_type or url.endswith('.mp3'):
                format_hint = 'mp3'
            elif 'ogg' in content_type or url.endswith('.ogg'):
                format_hint = 'ogg'
            elif 'flac' in content_type or url.endswith('.flac'):
                format_hint = 'flac'
            
            return load_audio_from_bytes(audio_bytes, target_sr, format_hint)


def convert_to_mono_16k(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    Конвертация аудио в mono 16kHz
    
    Args:
        audio: Исходный аудио массив
        original_sr: Исходная частота дискретизации
        target_sr: Целевая частота дискретизации
        
    Returns:
        Конвертированный аудио массив
    """
    import librosa
    
    # Конвертируем в mono если стерео
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Ресэмплинг
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    return audio.astype(np.float32)


def save_temp_audio(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    suffix: str = ".wav"
) -> str:
    """
    Сохранение аудио во временный файл
    
    Args:
        audio: Аудио массив
        sample_rate: Частота дискретизации
        suffix: Расширение файла
        
    Returns:
        Путь к временному файлу
    """
    import soundfile as sf
    
    # Создаём временный файл который не удаляется автоматически
    fd = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = fd.name
    fd.close()
    
    # Сохраняем аудио
    sf.write(temp_path, audio, sample_rate)
    
    return temp_path


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Получение длительности аудио в секундах
    
    Args:
        audio: Аудио массив
        sample_rate: Частота дискретизации
        
    Returns:
        Длительность в секундах
    """
    return len(audio) / sample_rate


class AudioBuffer:
    """
    Буфер для накопления аудио данных в streaming режиме
    
    Позволяет накапливать аудио и выдавать чанки фиксированной длины
    для обработки ASR моделью.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_duration_sec: float = 2.0,
        overlap_sec: float = 0.0
    ):
        """
        Args:
            sample_rate: Частота дискретизации входящего аудио
            chunk_duration_sec: Длина чанка для обработки в секундах
            overlap_sec: Перекрытие между чанками в секундах
        """
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        
        self.chunk_size = int(chunk_duration_sec * TARGET_SAMPLE_RATE)
        self.overlap_size = int(overlap_sec * TARGET_SAMPLE_RATE)
        
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
    
    def add_samples(self, samples: np.ndarray):
        """
        Добавление семплов в буфер
        
        Args:
            samples: Аудио семплы (float32, mono)
        """
        # Ресэмплинг если нужно
        if self.sample_rate != TARGET_SAMPLE_RATE:
            import librosa
            samples = librosa.resample(
                samples.astype(np.float32),
                orig_sr=self.sample_rate,
                target_sr=TARGET_SAMPLE_RATE
            )
        
        # Нормализация
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        # Добавляем в буфер
        self._buffer = np.concatenate([self._buffer, samples])
    
    def has_chunk(self) -> bool:
        """Проверка наличия готового чанка"""
        return len(self._buffer) >= self.chunk_size
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """
        Получение готового чанка
        
        Returns:
            Чанк аудио или None если недостаточно данных
        """
        if not self.has_chunk():
            return None
        
        # Извлекаем чанк
        chunk = self._buffer[:self.chunk_size].copy()
        
        # Сдвигаем буфер с учётом overlap
        shift = self.chunk_size - self.overlap_size
        self._buffer = self._buffer[shift:]
        
        return chunk
    
    def get_remaining(self) -> Optional[np.ndarray]:
        """
        Получение оставшихся данных в буфере
        
        Returns:
            Оставшиеся данные или None если буфер пуст
        """
        if len(self._buffer) == 0:
            return None
        
        remaining = self._buffer.copy()
        self._buffer = np.array([], dtype=np.float32)
        
        return remaining
    
    def clear(self):
        """Очистка буфера"""
        self._buffer = np.array([], dtype=np.float32)
    
    def get_buffer_duration(self) -> float:
        """Длительность данных в буфере в секундах"""
        return len(self._buffer) / TARGET_SAMPLE_RATE
    
    @property
    def buffer_size(self) -> int:
        """Количество семплов в буфере"""
        return len(self._buffer)


class AudioChunker:
    """
    Разбиение длинного аудио на чанки для параллельной обработки
    
    Используется для batch обработки длинных аудио файлов
    """
    
    def __init__(
        self,
        chunk_duration_sec: float = 30.0,
        overlap_sec: float = 1.0,
        min_chunk_duration_sec: float = 1.0
    ):
        """
        Args:
            chunk_duration_sec: Длина каждого чанка
            overlap_sec: Перекрытие между чанками для лучшего качества
            min_chunk_duration_sec: Минимальная длина последнего чанка
        """
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.min_chunk_duration_sec = min_chunk_duration_sec
    
    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE
    ) -> list[Tuple[np.ndarray, float, float]]:
        """
        Разбиение аудио на чанки
        
        Args:
            audio: Аудио массив
            sample_rate: Частота дискретизации
            
        Returns:
            Список кортежей (chunk_audio, start_time, end_time)
        """
        chunk_samples = int(self.chunk_duration_sec * sample_rate)
        overlap_samples = int(self.overlap_sec * sample_rate)
        min_samples = int(self.min_chunk_duration_sec * sample_rate)
        
        chunks = []
        position = 0
        
        while position < len(audio):
            # Определяем конец чанка
            end = min(position + chunk_samples, len(audio))
            
            # Если остаток слишком маленький, объединяем с предыдущим
            remaining = len(audio) - end
            if 0 < remaining < min_samples:
                end = len(audio)
            
            # Извлекаем чанк
            chunk = audio[position:end]
            start_time = position / sample_rate
            end_time = end / sample_rate
            
            chunks.append((chunk, start_time, end_time))
            
            # Сдвигаемся с учётом overlap
            position = end - overlap_samples
            
            # Если достигли конца, выходим
            if end >= len(audio):
                break
        
        return chunks
    
    def merge_transcriptions(
        self,
        transcriptions: list[Tuple[str, float, float]]
    ) -> str:
        """
        Объединение транскрипций чанков
        
        Args:
            transcriptions: Список (text, start_time, end_time)
            
        Returns:
            Объединённая транскрипция
        """
        # Простое объединение - можно улучшить для удаления дубликатов на границах
        return " ".join(t[0].strip() for t in transcriptions if t[0].strip())


if __name__ == "__main__":
    # Тестирование
    print("Тестирование AudioBuffer...")
    
    buffer = AudioBuffer(chunk_duration_sec=2.0)
    
    # Добавляем 3 секунды аудио
    test_audio = np.random.randn(48000).astype(np.float32)  # 3 секунды при 16kHz
    buffer.add_samples(test_audio)
    
    print(f"Размер буфера: {buffer.buffer_size} семплов")
    print(f"Длительность: {buffer.get_buffer_duration():.2f} сек")
    print(f"Есть чанк: {buffer.has_chunk()}")
    
    if buffer.has_chunk():
        chunk = buffer.get_chunk()
        print(f"Размер чанка: {len(chunk)} семплов")
        print(f"Остаток в буфере: {buffer.buffer_size} семплов")

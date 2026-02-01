"""
RunPod Handler - точка входа для ASR сервиса

Поддерживает два режима работы:
1. WebSocket - real-time streaming транскрипция
2. HTTP (RunPod job) - batch обработка файлов/URL

Протокол:
- При запуске job без input или с action="websocket" - запускается WebSocket сервер
- При запуске с input.audio или input.url - batch транскрипция
"""

import os
import sys
import json
import asyncio
import logging
import threading
from typing import Dict, Any, Optional
import tempfile

import runpod

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные для WebSocket сервера
websocket_server = None
websocket_thread = None


def get_connection_info() -> Dict[str, Any]:
    """Получение информации о подключении к воркеру"""
    public_ip = os.environ.get('RUNPOD_PUBLIC_IP', 'localhost')
    tcp_port = int(os.environ.get('RUNPOD_TCP_PORT_8765', '8765'))
    pod_id = os.environ.get('RUNPOD_POD_ID', 'unknown')
    
    return {
        "public_ip": public_ip,
        "tcp_port": tcp_port,
        "pod_id": pod_id,
        "websocket_url": f"ws://{public_ip}:{tcp_port}"
    }


async def start_websocket_server_async():
    """Асинхронный запуск WebSocket сервера"""
    from websocket_handler import WebSocketASRServer
    
    global websocket_server
    
    websocket_server = WebSocketASRServer(
        host="0.0.0.0",
        port=8765,
        max_connections=10
    )
    
    await websocket_server.start()


def run_websocket_in_thread():
    """Запуск WebSocket сервера в отдельном потоке"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(start_websocket_server_async())
    except Exception as e:
        logger.error(f"Ошибка WebSocket сервера: {e}")
    finally:
        loop.close()


def start_websocket_mode(job: Dict) -> Dict[str, Any]:
    """
    Запуск в режиме WebSocket сервера
    
    Сервер работает до получения команды shutdown через WebSocket
    или до timeout'а job'а
    """
    global websocket_thread
    
    connection_info = get_connection_info()
    logger.info(f"Запуск WebSocket режима: {connection_info}")
    
    # Отправляем информацию о подключении через progress
    runpod.serverless.progress_update(
        job,
        f"WebSocket сервер запущен: {connection_info['websocket_url']}"
    )
    
    # Запускаем WebSocket сервер в отдельном потоке
    websocket_thread = threading.Thread(target=run_websocket_in_thread)
    websocket_thread.start()
    
    # Ждём завершения сервера
    websocket_thread.join()
    
    return {
        "status": "completed",
        "mode": "websocket",
        "message": "WebSocket сервер завершил работу",
        **connection_info
    }


def process_batch_transcription(job: Dict) -> Dict[str, Any]:
    """
    Batch транскрипция аудио файла
    
    Поддерживаемые входные данные:
    - audio_base64: Base64 закодированное аудио
    - audio_url: URL аудио файла
    - audio_path: Путь к файлу (для локального тестирования)
    """
    from asr_engine import get_engine, TranscriptionResult
    from audio_utils import (
        load_audio_from_base64,
        load_audio_from_bytes,
        save_temp_audio,
        AudioChunker,
        get_audio_duration
    )
    import time
    
    job_input = job.get("input", {})
    
    # Получаем параметры
    timestamps = job_input.get("timestamps", True)
    use_local_attention = job_input.get("use_local_attention", False)
    auto_detect_long = job_input.get("auto_detect_long", True)
    chunk_long_audio = job_input.get("chunk_long_audio", False)
    chunk_duration = job_input.get("chunk_duration", 30.0)
    
    start_time = time.time()
    temp_files = []  # Для очистки
    
    try:
        # Загружаем аудио
        audio_path = None
        audio_array = None
        sample_rate = 16000
        
        if "audio_base64" in job_input:
            logger.info("Загрузка аудио из base64...")
            audio_array, sample_rate = load_audio_from_base64(
                job_input["audio_base64"],
                format_hint=job_input.get("format")
            )
            
        elif "audio_url" in job_input:
            logger.info(f"Загрузка аудио из URL: {job_input['audio_url']}")
            # Синхронная версия для batch режима
            import aiohttp
            import asyncio
            
            async def download():
                from audio_utils import load_audio_from_url
                return await load_audio_from_url(job_input["audio_url"])
            
            loop = asyncio.new_event_loop()
            audio_array, sample_rate = loop.run_until_complete(download())
            loop.close()
            
        elif "audio_path" in job_input:
            # Для локального тестирования
            audio_path = job_input["audio_path"]
            logger.info(f"Загрузка аудио из файла: {audio_path}")
            
        else:
            return {
                "error": "Не указан источник аудио. Используйте audio_base64, audio_url или audio_path"
            }
        
        # Если загрузили массив, сохраняем во временный файл
        if audio_array is not None:
            audio_path = save_temp_audio(audio_array, sample_rate)
            temp_files.append(audio_path)
            
            # Определяем длительность для автоматического выбора attention
            duration = get_audio_duration(audio_array, sample_rate)
            logger.info(f"Длительность аудио: {duration:.2f} сек")
            
            # Автоматически включаем local attention для длинных аудио
            if auto_detect_long and duration > 24 * 60:  # > 24 минут
                logger.info("Автоматически включаем local attention для длинного аудио")
                use_local_attention = True
        
        # Получаем движок
        engine = get_engine()
        
        # Транскрипция
        if chunk_long_audio and audio_array is not None:
            # Разбиваем на чанки для параллельной обработки
            logger.info(f"Разбиение на чанки по {chunk_duration} сек...")
            
            chunker = AudioChunker(
                chunk_duration_sec=chunk_duration,
                overlap_sec=1.0
            )
            chunks = chunker.chunk_audio(audio_array, sample_rate)
            
            logger.info(f"Создано {len(chunks)} чанков")
            
            # Обрабатываем чанки
            transcriptions = []
            for i, (chunk_audio, start_t, end_t) in enumerate(chunks):
                logger.info(f"Обработка чанка {i+1}/{len(chunks)}: {start_t:.1f}s - {end_t:.1f}s")
                
                # Сохраняем чанк
                chunk_path = save_temp_audio(chunk_audio, sample_rate)
                temp_files.append(chunk_path)
                
                # Транскрипция
                result = engine.transcribe(
                    [chunk_path],
                    timestamps=timestamps,
                    use_local_attention=use_local_attention
                )[0]
                
                transcriptions.append((result.text, start_t, end_t))
            
            # Объединяем результаты
            final_text = chunker.merge_transcriptions(transcriptions)
            
            result = TranscriptionResult(
                text=final_text,
                processing_time=time.time() - start_time
            )
            
        else:
            # Обычная транскрипция
            results = engine.transcribe(
                [audio_path],
                timestamps=timestamps,
                use_local_attention=use_local_attention
            )
            result = results[0]
        
        processing_time = time.time() - start_time
        
        # Формируем ответ
        response = {
            "status": "success",
            "text": result.text,
            "processing_time_sec": round(processing_time, 3)
        }
        
        if timestamps and result.word_timestamps:
            response["word_timestamps"] = result.word_timestamps
        
        if timestamps and result.segment_timestamps:
            response["segment_timestamps"] = result.segment_timestamps
        
        # Информация о модели
        response["model_info"] = engine.get_model_info()
        
        logger.info(f"Транскрипция завершена за {processing_time:.2f} сек")
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
    finally:
        # Очистка временных файлов
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


def handler(job: Dict) -> Dict[str, Any]:
    """
    Главный обработчик RunPod job'ов
    
    Режимы:
    1. WebSocket (action="websocket" или пустой input): запускает WebSocket сервер
    2. Batch (audio_base64/audio_url): транскрипция файла
    
    Примеры input:
    
    WebSocket режим:
    {"action": "websocket"}
    или
    {}
    
    Batch режим:
    {
        "audio_base64": "...",
        "timestamps": true,
        "use_local_attention": false
    }
    или
    {
        "audio_url": "https://example.com/audio.wav",
        "timestamps": true
    }
    """
    logger.info(f"Получен job: {job.get('id', 'unknown')}")
    
    job_input = job.get("input", {})
    
    # Определяем режим работы
    action = job_input.get("action", "").lower()
    
    if action == "websocket" or not job_input:
        # WebSocket режим
        logger.info("Запуск в режиме WebSocket")
        return start_websocket_mode(job)
        
    elif action == "info":
        # Информация о сервисе
        from asr_engine import get_engine
        engine = get_engine()
        
        return {
            "status": "success",
            "model_info": engine.get_model_info(),
            "connection_info": get_connection_info(),
            "supported_actions": ["websocket", "info", "transcribe"],
            "supported_formats": ["wav", "mp3", "flac", "ogg", "webm"]
        }
        
    elif "audio_base64" in job_input or "audio_url" in job_input or "audio_path" in job_input:
        # Batch транскрипция
        logger.info("Запуск batch транскрипции")
        return process_batch_transcription(job)
        
    else:
        return {
            "status": "error",
            "error": "Неизвестное действие. Используйте action='websocket' или предоставьте audio_base64/audio_url",
            "supported_actions": ["websocket", "info"],
            "example_batch": {
                "audio_url": "https://example.com/audio.wav",
                "timestamps": True
            },
            "example_websocket": {
                "action": "websocket"
            }
        }


# Генератор для streaming ответов (для будущего использования)
async def async_generator_handler(job: Dict):
    """
    Асинхронный генератор для streaming ответов
    (Для будущей поддержки RunPod streaming)
    """
    job_input = job.get("input", {})
    
    # Пока просто возвращаем результат
    result = handler(job)
    yield result


if __name__ == "__main__":
    logger.info("Запуск Parakeet ASR сервиса...")
    logger.info(f"Connection info: {get_connection_info()}")
    
    # Предзагрузка модели при старте
    logger.info("Предзагрузка ASR модели...")
    from asr_engine import get_engine
    engine = get_engine()
    logger.info(f"Модель загружена: {engine.get_model_info()}")
    
    # Запуск RunPod serverless
    runpod.serverless.start({
        "handler": handler,
        # "return_aggregate_stream": True  # Для будущей поддержки streaming
    })

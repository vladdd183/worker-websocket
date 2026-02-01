"""
WebSocket Handler для real-time транскрипции аудио

Протокол:
1. Клиент подключается к WebSocket
2. Отправляет JSON с конфигурацией: {"action": "start", "sample_rate": 16000}
3. Стримит бинарные аудио чанки
4. Получает JSON с транскрипцией: {"text": "...", "is_final": true/false}
5. Отправляет {"action": "stop"} для завершения
"""

import asyncio
import json
import logging
import numpy as np
import io
import struct
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import websockets
from websockets.server import WebSocketServerProtocol

from asr_engine import get_engine, TranscriptionResult
from audio_utils import AudioBuffer, convert_to_mono_16k

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientState(Enum):
    """Состояние клиента"""
    CONNECTED = "connected"
    STREAMING = "streaming"
    PROCESSING = "processing"
    DISCONNECTED = "disconnected"


@dataclass
class ClientSession:
    """Сессия клиента WebSocket"""
    websocket: WebSocketServerProtocol
    state: ClientState = ClientState.CONNECTED
    sample_rate: int = 16000
    audio_buffer: Optional['AudioBuffer'] = None
    chunk_duration_sec: float = 2.0  # Размер чанка для обработки
    min_audio_length_sec: float = 0.5  # Минимальная длина для транскрипции
    include_timestamps: bool = True
    accumulated_text: str = ""
    
    def __post_init__(self):
        from audio_utils import AudioBuffer
        self.audio_buffer = AudioBuffer(
            sample_rate=self.sample_rate,
            chunk_duration_sec=self.chunk_duration_sec
        )


class WebSocketASRServer:
    """
    WebSocket сервер для real-time ASR
    
    Поддерживает:
    - Streaming транскрипцию с низкой latency
    - Множественные одновременные подключения
    - Конфигурацию через JSON сообщения
    - Graceful shutdown
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_connections: int = 10
    ):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.sessions: Dict[str, ClientSession] = {}
        self.server: Optional[websockets.WebSocketServer] = None
        self._shutdown_event = asyncio.Event()
        
        # Загружаем ASR движок
        logger.info("Инициализация ASR движка...")
        self.engine = get_engine()
        logger.info("ASR движок готов")
    
    async def start(self):
        """Запуск WebSocket сервера"""
        logger.info(f"Запуск WebSocket сервера на {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB макс размер сообщения
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info(f"WebSocket сервер запущен на ws://{self.host}:{self.port}")
        
        # Ждём сигнала завершения
        await self._shutdown_event.wait()
        
        # Закрываем сервер
        self.server.close()
        await self.server.wait_closed()
        logger.info("WebSocket сервер остановлен")
    
    async def stop(self):
        """Остановка сервера"""
        logger.info("Получен сигнал остановки...")
        self._shutdown_event.set()
    
    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Обработка подключения клиента"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Новое подключение: {client_id}")
        
        # Проверяем лимит подключений
        if len(self.sessions) >= self.max_connections:
            await websocket.close(1013, "Сервер перегружен")
            logger.warning(f"Отклонено подключение {client_id}: превышен лимит")
            return
        
        # Создаём сессию
        session = ClientSession(websocket=websocket)
        self.sessions[client_id] = session
        
        try:
            await self._client_loop(client_id, session)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Клиент {client_id} отключился: {e.code} {e.reason}")
        except Exception as e:
            logger.error(f"Ошибка в сессии {client_id}: {e}")
            await self._send_error(websocket, str(e))
        finally:
            # Очищаем сессию
            if client_id in self.sessions:
                del self.sessions[client_id]
            logger.info(f"Сессия {client_id} завершена")
    
    async def _client_loop(self, client_id: str, session: ClientSession):
        """Основной цикл обработки клиента"""
        websocket = session.websocket
        
        # Отправляем приветственное сообщение
        await self._send_json(websocket, {
            "type": "welcome",
            "message": "Подключено к Parakeet ASR",
            "model": "parakeet-tdt-0.6b-v3",
            "supported_sample_rates": [16000, 44100, 48000],
            "commands": ["start", "stop", "config", "ping"]
        })
        
        async for message in websocket:
            if isinstance(message, str):
                # JSON команда
                await self._handle_json_message(client_id, session, message)
            elif isinstance(message, bytes):
                # Бинарные аудио данные
                await self._handle_audio_data(client_id, session, message)
    
    async def _handle_json_message(
        self, 
        client_id: str, 
        session: ClientSession, 
        message: str
    ):
        """Обработка JSON команд от клиента"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await self._send_error(session.websocket, "Неверный JSON")
            return
        
        action = data.get("action", "").lower()
        
        if action == "start":
            # Начало стриминга
            session.sample_rate = data.get("sample_rate", 16000)
            session.chunk_duration_sec = data.get("chunk_duration", 2.0)
            session.include_timestamps = data.get("timestamps", True)
            session.state = ClientState.STREAMING
            session.accumulated_text = ""
            
            # Пересоздаём буфер с новыми настройками
            from audio_utils import AudioBuffer
            session.audio_buffer = AudioBuffer(
                sample_rate=session.sample_rate,
                chunk_duration_sec=session.chunk_duration_sec
            )
            
            await self._send_json(session.websocket, {
                "type": "started",
                "sample_rate": session.sample_rate,
                "chunk_duration": session.chunk_duration_sec
            })
            logger.info(f"Клиент {client_id} начал стриминг")
            
        elif action == "stop":
            # Конец стриминга - обрабатываем остаток буфера
            if session.state == ClientState.STREAMING:
                await self._process_final(client_id, session)
            
            session.state = ClientState.CONNECTED
            await self._send_json(session.websocket, {
                "type": "stopped",
                "final_text": session.accumulated_text
            })
            logger.info(f"Клиент {client_id} остановил стриминг")
            
        elif action == "config":
            # Обновление конфигурации
            if "chunk_duration" in data:
                session.chunk_duration_sec = data["chunk_duration"]
            if "timestamps" in data:
                session.include_timestamps = data["timestamps"]
            
            await self._send_json(session.websocket, {
                "type": "config_updated",
                "config": {
                    "chunk_duration": session.chunk_duration_sec,
                    "timestamps": session.include_timestamps
                }
            })
            
        elif action == "ping":
            await self._send_json(session.websocket, {"type": "pong"})
            
        elif action == "shutdown":
            # Команда остановки сервера (только для администратора)
            logger.info(f"Получена команда shutdown от {client_id}")
            await self._send_json(session.websocket, {
                "type": "shutdown_acknowledged"
            })
            await self.stop()
            
        elif action == "info":
            # Информация о модели
            info = self.engine.get_model_info()
            await self._send_json(session.websocket, {
                "type": "info",
                "model_info": info
            })
            
        else:
            await self._send_error(
                session.websocket, 
                f"Неизвестная команда: {action}"
            )
    
    async def _handle_audio_data(
        self, 
        client_id: str, 
        session: ClientSession, 
        data: bytes
    ):
        """Обработка бинарных аудио данных"""
        if session.state != ClientState.STREAMING:
            await self._send_error(
                session.websocket, 
                "Стриминг не начат. Отправьте {\"action\": \"start\"}"
            )
            return
        
        # Декодируем аудио данные
        try:
            # Предполагаем PCM16 Little Endian
            audio_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_samples /= 32768.0  # Нормализация
        except Exception as e:
            await self._send_error(session.websocket, f"Ошибка декодирования аудио: {e}")
            return
        
        # Добавляем в буфер
        session.audio_buffer.add_samples(audio_samples)
        
        # Проверяем, есть ли готовый чанк для обработки
        if session.audio_buffer.has_chunk():
            chunk = session.audio_buffer.get_chunk()
            await self._process_chunk(client_id, session, chunk)
    
    async def _process_chunk(
        self, 
        client_id: str, 
        session: ClientSession, 
        audio_chunk: np.ndarray
    ):
        """Обработка аудио чанка через ASR"""
        session.state = ClientState.PROCESSING
        
        try:
            # Транскрипция
            result = self.engine.transcribe_audio_array(
                audio_chunk,
                sample_rate=16000,  # Буфер уже нормализует до 16kHz
                timestamps=session.include_timestamps
            )
            
            # Обновляем накопленный текст
            if result.text:
                session.accumulated_text += " " + result.text
                session.accumulated_text = session.accumulated_text.strip()
            
            # Отправляем результат
            response = {
                "type": "transcription",
                "text": result.text,
                "is_final": False,
                "accumulated_text": session.accumulated_text
            }
            
            if session.include_timestamps and result.word_timestamps:
                response["word_timestamps"] = result.word_timestamps
            
            if result.processing_time:
                response["processing_time_ms"] = round(result.processing_time * 1000, 2)
            
            await self._send_json(session.websocket, response)
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции для {client_id}: {e}")
            await self._send_error(session.websocket, f"Ошибка транскрипции: {e}")
        
        finally:
            session.state = ClientState.STREAMING
    
    async def _process_final(self, client_id: str, session: ClientSession):
        """Обработка оставшихся данных в буфере"""
        remaining = session.audio_buffer.get_remaining()
        
        if remaining is not None and len(remaining) > 0:
            # Минимальная длина для обработки
            min_samples = int(session.min_audio_length_sec * 16000)
            
            if len(remaining) >= min_samples:
                await self._process_chunk(client_id, session, remaining)
    
    async def _send_json(self, websocket: WebSocketServerProtocol, data: dict):
        """Отправка JSON сообщения"""
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Ошибка отправки JSON: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, message: str):
        """Отправка сообщения об ошибке"""
        await self._send_json(websocket, {
            "type": "error",
            "message": message
        })


async def run_websocket_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    shutdown_callback: Optional[Callable] = None
) -> WebSocketASRServer:
    """
    Запуск WebSocket сервера
    
    Args:
        host: Хост для прослушивания
        port: Порт для прослушивания
        shutdown_callback: Callback при завершении
        
    Returns:
        WebSocketASRServer instance
    """
    server = WebSocketASRServer(host=host, port=port)
    
    try:
        await server.start()
    finally:
        if shutdown_callback:
            shutdown_callback()
    
    return server


if __name__ == "__main__":
    # Тестовый запуск сервера
    asyncio.run(run_websocket_server())

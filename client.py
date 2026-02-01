"""
Клиент для тестирования Parakeet ASR сервиса

Поддерживает:
1. WebSocket real-time стриминг с микрофона или файла
2. HTTP batch транскрипцию через RunPod API
"""

import asyncio
import json
import base64
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import websockets
import aiohttp


class ASRWebSocketClient:
    """Клиент для WebSocket real-time транскрипции"""
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.websocket = None
        self.is_running = False
    
    async def connect(self):
        """Подключение к WebSocket серверу"""
        print(f"Подключение к {self.websocket_url}...")
        self.websocket = await websockets.connect(
            self.websocket_url,
            max_size=10 * 1024 * 1024
        )
        
        # Получаем приветственное сообщение
        response = await self.websocket.recv()
        data = json.loads(response)
        print(f"Подключено: {data}")
        
        return data
    
    async def start_streaming(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        timestamps: bool = True
    ):
        """Начать streaming сессию"""
        await self.websocket.send(json.dumps({
            "action": "start",
            "sample_rate": sample_rate,
            "chunk_duration": chunk_duration,
            "timestamps": timestamps
        }))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        print(f"Streaming начат: {data}")
        
        self.is_running = True
        return data
    
    async def send_audio_chunk(self, audio_bytes: bytes):
        """Отправить чанк аудио"""
        if not self.is_running:
            raise RuntimeError("Streaming не запущен. Вызовите start_streaming()")
        
        await self.websocket.send(audio_bytes)
    
    async def receive_transcription(self) -> dict:
        """Получить транскрипцию"""
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def stop_streaming(self):
        """Остановить streaming"""
        await self.websocket.send(json.dumps({"action": "stop"}))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        print(f"Streaming остановлен: {data}")
        
        self.is_running = False
        return data
    
    async def get_info(self) -> dict:
        """Получить информацию о модели"""
        await self.websocket.send(json.dumps({"action": "info"}))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def shutdown_server(self):
        """Отправить команду shutdown серверу"""
        await self.websocket.send(json.dumps({"action": "shutdown"}))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def close(self):
        """Закрыть соединение"""
        if self.websocket:
            await self.websocket.close()


async def stream_file_to_websocket(
    websocket_url: str,
    audio_file: str,
    chunk_duration: float = 2.0,
    sample_rate: int = 16000
):
    """
    Стриминг аудио файла через WebSocket
    
    Args:
        websocket_url: URL WebSocket сервера
        audio_file: Путь к аудио файлу
        chunk_duration: Длительность чанка в секундах
        sample_rate: Частота дискретизации
    """
    import soundfile as sf
    import numpy as np
    
    # Загружаем аудио
    print(f"Загрузка файла: {audio_file}")
    audio, sr = sf.read(audio_file)
    
    # Конвертируем в mono если стерео
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Ресэмплинг если нужно
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    
    # Конвертируем в int16 PCM
    audio = (audio * 32767).astype(np.int16)
    
    print(f"Длительность: {len(audio) / sample_rate:.2f} сек")
    
    # Подключаемся
    client = ASRWebSocketClient(websocket_url)
    await client.connect()
    
    # Начинаем streaming
    await client.start_streaming(
        sample_rate=sample_rate,
        chunk_duration=chunk_duration
    )
    
    # Размер чанка в семплах
    chunk_samples = int(chunk_duration * sample_rate)
    
    # Отправляем чанки
    accumulated_text = ""
    
    async def receive_loop():
        nonlocal accumulated_text
        try:
            while client.is_running:
                response = await asyncio.wait_for(
                    client.receive_transcription(),
                    timeout=30.0
                )
                
                if response.get("type") == "transcription":
                    text = response.get("text", "")
                    if text:
                        print(f"[{response.get('processing_time_ms', 0):.0f}ms] {text}")
                    accumulated_text = response.get("accumulated_text", accumulated_text)
                elif response.get("type") == "error":
                    print(f"Ошибка: {response.get('message')}")
                    
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed:
            pass
    
    # Запускаем получение в фоне
    receive_task = asyncio.create_task(receive_loop())
    
    try:
        # Отправляем чанки
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            
            # Конвертируем в bytes
            audio_bytes = chunk.tobytes()
            
            await client.send_audio_chunk(audio_bytes)
            
            # Небольшая задержка для имитации real-time
            await asyncio.sleep(chunk_duration * 0.5)
        
        # Ждём последние ответы
        await asyncio.sleep(2.0)
        
    finally:
        # Останавливаем streaming
        result = await client.stop_streaming()
        receive_task.cancel()
        
        print("\n" + "=" * 50)
        print("Финальная транскрипция:")
        print(result.get("final_text", accumulated_text))
        print("=" * 50)
        
        await client.close()


async def batch_transcribe_runpod(
    endpoint_url: str,
    api_key: str,
    audio_file: Optional[str] = None,
    audio_url: Optional[str] = None,
    timestamps: bool = True
):
    """
    Batch транскрипция через RunPod API
    
    Args:
        endpoint_url: URL RunPod endpoint
        api_key: RunPod API ключ
        audio_file: Путь к локальному файлу
        audio_url: URL аудио файла
        timestamps: Включить timestamps
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Формируем input
    job_input = {"timestamps": timestamps}
    
    if audio_file:
        print(f"Загрузка файла: {audio_file}")
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        job_input["audio_base64"] = base64.b64encode(audio_bytes).decode()
        
        # Определяем формат
        suffix = Path(audio_file).suffix.lower()
        if suffix in [".mp3", ".ogg", ".flac", ".webm"]:
            job_input["format"] = suffix[1:]
            
    elif audio_url:
        job_input["audio_url"] = audio_url
    else:
        raise ValueError("Укажите audio_file или audio_url")
    
    async with aiohttp.ClientSession() as session:
        # Отправляем job
        print("Отправка запроса...")
        
        async with session.post(
            f"{endpoint_url}/run",
            headers=headers,
            json={"input": job_input}
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Ошибка API: {response.status} - {text}")
            
            result = await response.json()
            job_id = result.get("id")
            print(f"Job создан: {job_id}")
        
        # Ждём результат
        print("Ожидание результата...")
        
        while True:
            async with session.get(
                f"{endpoint_url}/status/{job_id}",
                headers=headers
            ) as response:
                result = await response.json()
                status = result.get("status")
                
                if status == "COMPLETED":
                    output = result.get("output", {})
                    print("\n" + "=" * 50)
                    print("Транскрипция:")
                    print(output.get("text", ""))
                    print("=" * 50)
                    print(f"Время обработки: {output.get('processing_time_sec', 0):.2f} сек")
                    
                    if timestamps and output.get("segment_timestamps"):
                        print("\nСегменты:")
                        for seg in output["segment_timestamps"][:10]:
                            print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['segment']}")
                    
                    return output
                    
                elif status == "FAILED":
                    error = result.get("error", "Unknown error")
                    raise RuntimeError(f"Job failed: {error}")
                    
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    print(f"  Статус: {status}...")
                    await asyncio.sleep(2)
                    
                else:
                    print(f"  Неизвестный статус: {status}")
                    await asyncio.sleep(2)


async def demo_websocket(websocket_url: str):
    """Демо WebSocket подключения"""
    client = ASRWebSocketClient(websocket_url)
    
    try:
        await client.connect()
        
        # Получаем информацию
        info = await client.get_info()
        print(f"\nИнформация о модели:")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        
        print("\nДля отправки аудио используйте stream_file_to_websocket()")
        
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Клиент для Parakeet ASR сервиса"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команда")
    
    # WebSocket streaming
    ws_parser = subparsers.add_parser("stream", help="WebSocket streaming файла")
    ws_parser.add_argument("--url", required=True, help="WebSocket URL (ws://host:port)")
    ws_parser.add_argument("--file", required=True, help="Путь к аудио файлу")
    ws_parser.add_argument("--chunk", type=float, default=2.0, help="Длительность чанка (сек)")
    
    # Batch транскрипция
    batch_parser = subparsers.add_parser("batch", help="Batch транскрипция через RunPod")
    batch_parser.add_argument("--endpoint", required=True, help="RunPod endpoint URL")
    batch_parser.add_argument("--api-key", required=True, help="RunPod API ключ")
    batch_parser.add_argument("--file", help="Путь к аудио файлу")
    batch_parser.add_argument("--url", help="URL аудио файла")
    batch_parser.add_argument("--no-timestamps", action="store_true", help="Без timestamps")
    
    # Демо
    demo_parser = subparsers.add_parser("demo", help="Демо подключения")
    demo_parser.add_argument("--url", required=True, help="WebSocket URL")
    
    args = parser.parse_args()
    
    if args.command == "stream":
        asyncio.run(stream_file_to_websocket(
            websocket_url=args.url,
            audio_file=args.file,
            chunk_duration=args.chunk
        ))
        
    elif args.command == "batch":
        if not args.file and not args.url:
            print("Укажите --file или --url")
            sys.exit(1)
            
        asyncio.run(batch_transcribe_runpod(
            endpoint_url=args.endpoint,
            api_key=args.api_key,
            audio_file=args.file,
            audio_url=args.url,
            timestamps=not args.no_timestamps
        ))
        
    elif args.command == "demo":
        asyncio.run(demo_websocket(args.url))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

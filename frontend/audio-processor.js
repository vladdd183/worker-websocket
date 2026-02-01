/**
 * Audio Processor - работа с микрофоном и FFmpeg WASM
 */

// ============================================
// Microphone Recorder
// ============================================

class MicrophoneRecorder {
    constructor(options = {}) {
        this.sampleRate = options.sampleRate || 16000;
        this.chunkSizeMs = options.chunkSizeMs || 500;
        this.onChunk = options.onChunk || (() => {});
        this.onVisualizerData = options.onVisualizerData || (() => {});
        
        this.mediaStream = null;
        this.audioContext = null;
        this.processor = null;
        this.analyser = null;
        this.isRecording = false;
        
        this.audioBuffer = [];
        this.chunkSamples = Math.floor((this.chunkSizeMs / 1000) * this.sampleRate);
    }
    
    async start() {
        try {
            // Запрашиваем доступ к микрофону
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            // Создаём AudioContext
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            // Источник из микрофона
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Анализатор для визуализации
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
            
            // ScriptProcessor для получения сырых данных
            // (AudioWorklet был бы лучше, но требует отдельного файла)
            const bufferSize = 4096;
            this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Добавляем в буфер
                this.audioBuffer.push(...inputData);
                
                // Проверяем, накопился ли чанк
                while (this.audioBuffer.length >= this.chunkSamples) {
                    const chunk = this.audioBuffer.splice(0, this.chunkSamples);
                    this.sendChunk(chunk);
                }
            };
            
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            
            // Запускаем визуализацию
            this.startVisualization();
            
            return true;
        } catch (error) {
            console.error('Ошибка доступа к микрофону:', error);
            throw error;
        }
    }
    
    stop() {
        this.isRecording = false;
        
        // Отправляем оставшиеся данные
        if (this.audioBuffer.length > 0) {
            this.sendChunk(this.audioBuffer);
            this.audioBuffer = [];
        }
        
        // Останавливаем медиапоток
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        // Закрываем AudioContext
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.processor = null;
        this.analyser = null;
    }
    
    sendChunk(samples) {
        // Конвертируем в Int16
        const int16Array = new Int16Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Вызываем callback
        this.onChunk(int16Array.buffer, samples.length / this.sampleRate);
    }
    
    startVisualization() {
        const draw = () => {
            if (!this.isRecording || !this.analyser) return;
            
            const bufferLength = this.analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            this.analyser.getByteFrequencyData(dataArray);
            
            this.onVisualizerData(dataArray);
            
            requestAnimationFrame(draw);
        };
        
        draw();
    }
    
    setChunkSize(ms) {
        this.chunkSizeMs = ms;
        this.chunkSamples = Math.floor((this.chunkSizeMs / 1000) * this.sampleRate);
    }
}


// ============================================
// FFmpeg WASM Audio Processor
// ============================================

class FFmpegProcessor {
    constructor() {
        this.ffmpeg = null;
        this.loaded = false;
        this.loading = false;
    }
    
    async load(onProgress = () => {}) {
        if (this.loaded) return;
        if (this.loading) {
            // Ждём загрузки
            while (this.loading) {
                await new Promise(r => setTimeout(r, 100));
            }
            return;
        }
        
        this.loading = true;
        
        try {
            const { FFmpeg } = FFmpegWASM;
            this.ffmpeg = new FFmpeg();
            
            this.ffmpeg.on('progress', ({ progress }) => {
                onProgress(Math.round(progress * 100));
            });
            
            // Загружаем core
            await this.ffmpeg.load({
                coreURL: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js',
                wasmURL: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm',
            });
            
            this.loaded = true;
        } finally {
            this.loading = false;
        }
    }
    
    async getAudioInfo(file) {
        await this.load();
        
        const { fetchFile } = FFmpegUtil;
        const inputName = 'input' + this.getExtension(file.name);
        
        await this.ffmpeg.writeFile(inputName, await fetchFile(file));
        
        // Получаем информацию через ffprobe-подобную команду
        // FFmpeg WASM не поддерживает ffprobe, так что используем другой подход
        
        // Создаём audio element для получения длительности
        const audioUrl = URL.createObjectURL(file);
        const audio = new Audio(audioUrl);
        
        return new Promise((resolve, reject) => {
            audio.onloadedmetadata = () => {
                URL.revokeObjectURL(audioUrl);
                resolve({
                    duration: audio.duration,
                    size: file.size
                });
            };
            audio.onerror = reject;
        });
    }
    
    async splitIntoChunks(file, chunkDurationSec, overlapSec = 0, onProgress = () => {}) {
        await this.load();
        
        const { fetchFile } = FFmpegUtil;
        const inputName = 'input' + this.getExtension(file.name);
        
        // Записываем входной файл
        await this.ffmpeg.writeFile(inputName, await fetchFile(file));
        
        // Получаем длительность
        const info = await this.getAudioInfo(file);
        const totalDuration = info.duration;
        
        const chunks = [];
        let position = 0;
        let chunkIndex = 0;
        
        while (position < totalDuration) {
            const startTime = position;
            const endTime = Math.min(position + chunkDurationSec, totalDuration);
            const duration = endTime - startTime;
            
            // Пропускаем слишком короткие чанки
            if (duration < 0.5 && chunks.length > 0) break;
            
            const outputName = `chunk_${chunkIndex}.wav`;
            
            // Конвертируем чанк в WAV 16kHz mono
            await this.ffmpeg.exec([
                '-ss', startTime.toString(),
                '-i', inputName,
                '-t', duration.toString(),
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                outputName
            ]);
            
            // Читаем результат
            const data = await this.ffmpeg.readFile(outputName);
            
            chunks.push({
                index: chunkIndex,
                startTime: startTime,
                endTime: endTime,
                duration: duration,
                data: new Blob([data.buffer], { type: 'audio/wav' })
            });
            
            // Удаляем временный файл
            await this.ffmpeg.deleteFile(outputName);
            
            // Прогресс
            onProgress(Math.round((endTime / totalDuration) * 100), chunkIndex);
            
            // Сдвигаемся с учётом overlap
            position = endTime - overlapSec;
            chunkIndex++;
        }
        
        // Удаляем входной файл
        await this.ffmpeg.deleteFile(inputName);
        
        return {
            chunks,
            totalDuration,
            chunkCount: chunks.length
        };
    }
    
    async convertToWav(file, onProgress = () => {}) {
        await this.load();
        
        const { fetchFile } = FFmpegUtil;
        const inputName = 'input' + this.getExtension(file.name);
        const outputName = 'output.wav';
        
        // Записываем входной файл
        await this.ffmpeg.writeFile(inputName, await fetchFile(file));
        
        // Конвертируем в WAV 16kHz mono
        await this.ffmpeg.exec([
            '-i', inputName,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            outputName
        ]);
        
        // Читаем результат
        const data = await this.ffmpeg.readFile(outputName);
        
        // Очистка
        await this.ffmpeg.deleteFile(inputName);
        await this.ffmpeg.deleteFile(outputName);
        
        return new Blob([data.buffer], { type: 'audio/wav' });
    }
    
    getExtension(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        return '.' + (ext || 'wav');
    }
}


// ============================================
// Audio Visualizer
// ============================================

class AudioVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.gradient = null;
        this.setupGradient();
    }
    
    setupGradient() {
        this.gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        this.gradient.addColorStop(0, '#10b981');
        this.gradient.addColorStop(0.5, '#6366f1');
        this.gradient.addColorStop(1, '#ef4444');
    }
    
    draw(dataArray) {
        const { width, height } = this.canvas;
        const bufferLength = dataArray.length;
        const barWidth = (width / bufferLength) * 2.5;
        
        // Очищаем
        this.ctx.fillStyle = '#0f172a';
        this.ctx.fillRect(0, 0, width, height);
        
        // Рисуем бары
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * height;
            
            this.ctx.fillStyle = this.gradient;
            this.ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            
            x += barWidth + 1;
        }
    }
    
    clear() {
        this.ctx.fillStyle = '#0f172a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
}


// ============================================
// Latency Chart
// ============================================

class LatencyChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = [];
        this.maxPoints = 100;
        this.maxLatency = 1000; // Автомасштабирование
    }
    
    addPoint(latency) {
        this.data.push({
            time: Date.now(),
            latency: latency
        });
        
        // Ограничиваем количество точек
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }
        
        // Автомасштабирование
        const maxInData = Math.max(...this.data.map(d => d.latency));
        this.maxLatency = Math.max(100, Math.ceil(maxInData / 100) * 100 + 100);
        
        this.draw();
    }
    
    draw() {
        const { width, height } = this.canvas;
        const padding = 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;
        
        // Очищаем
        this.ctx.fillStyle = '#0f172a';
        this.ctx.fillRect(0, 0, width, height);
        
        if (this.data.length < 2) return;
        
        // Оси
        this.ctx.strokeStyle = '#475569';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(padding, padding);
        this.ctx.lineTo(padding, height - padding);
        this.ctx.lineTo(width - padding, height - padding);
        this.ctx.stroke();
        
        // Подписи оси Y
        this.ctx.fillStyle = '#94a3b8';
        this.ctx.font = '11px monospace';
        this.ctx.textAlign = 'right';
        
        const ySteps = 5;
        for (let i = 0; i <= ySteps; i++) {
            const y = padding + (chartHeight / ySteps) * i;
            const value = Math.round(this.maxLatency - (this.maxLatency / ySteps) * i);
            this.ctx.fillText(value + 'ms', padding - 5, y + 4);
            
            // Горизонтальные линии
            this.ctx.strokeStyle = '#334155';
            this.ctx.beginPath();
            this.ctx.moveTo(padding, y);
            this.ctx.lineTo(width - padding, y);
            this.ctx.stroke();
        }
        
        // Рисуем линию
        this.ctx.strokeStyle = '#6366f1';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        this.data.forEach((point, i) => {
            const x = padding + (chartWidth / (this.maxPoints - 1)) * i;
            const y = padding + chartHeight - (point.latency / this.maxLatency) * chartHeight;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });
        
        this.ctx.stroke();
        
        // Точки
        this.ctx.fillStyle = '#6366f1';
        this.data.forEach((point, i) => {
            const x = padding + (chartWidth / (this.maxPoints - 1)) * i;
            const y = padding + chartHeight - (point.latency / this.maxLatency) * chartHeight;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Средняя линия
        if (this.data.length > 0) {
            const avg = this.data.reduce((sum, d) => sum + d.latency, 0) / this.data.length;
            const avgY = padding + chartHeight - (avg / this.maxLatency) * chartHeight;
            
            this.ctx.strokeStyle = '#10b981';
            this.ctx.lineWidth = 1;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(padding, avgY);
            this.ctx.lineTo(width - padding, avgY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            this.ctx.fillStyle = '#10b981';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(`avg: ${Math.round(avg)}ms`, padding + 5, avgY - 5);
        }
    }
    
    clear() {
        this.data = [];
        this.ctx.fillStyle = '#0f172a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
}


// ============================================
// Utilities
// ============================================

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)} сек`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} мин ${Math.floor(seconds % 60)} сек`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours} ч ${mins} мин`;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function calculateRTF(audioDuration, processingTime) {
    if (audioDuration <= 0) return '-';
    return (processingTime / audioDuration).toFixed(2) + 'x';
}


// Экспорт для глобального использования
window.MicrophoneRecorder = MicrophoneRecorder;
window.FFmpegProcessor = FFmpegProcessor;
window.AudioVisualizer = AudioVisualizer;
window.LatencyChart = LatencyChart;
window.AudioUtils = {
    formatTime,
    formatDuration,
    formatFileSize,
    calculateRTF
};

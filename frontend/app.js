/**
 * Parakeet ASR Frontend - Main Application
 */

// ============================================
// Global State
// ============================================

const state = {
    // Connection
    websocket: null,
    connected: false,
    
    // RunPod
    runpodJobId: null,
    runpodWorkerRunning: false,
    runpodPollingInterval: null,
    
    // Recording
    recorder: null,
    recording: false,
    recordingStartTime: null,
    
    // Stats
    stats: {
        chunksSent: 0,
        responsesReceived: 0,
        latencies: [],
        totalAudioDuration: 0,
        totalProcessingTime: 0
    },
    
    // Batch
    currentFile: null,
    ffmpeg: null,
    batchProcessing: false,
    batchChunks: [],
    
    // Pending chunks (для измерения latency)
    pendingChunks: new Map()
};


// ============================================
// DOM Elements
// ============================================

const elements = {
    // RunPod
    runpodApiKey: document.getElementById('runpod-api-key'),
    runpodEndpointId: document.getElementById('runpod-endpoint-id'),
    launchWorkerBtn: document.getElementById('launch-worker-btn'),
    stopWorkerBtn: document.getElementById('stop-worker-btn'),
    workerStatus: document.getElementById('worker-status'),
    
    // Connection
    wsUrl: document.getElementById('ws-url'),
    connectBtn: document.getElementById('connect-btn'),
    disconnectBtn: document.getElementById('disconnect-btn'),
    healthcheckBtn: document.getElementById('healthcheck-btn'),
    connectionStatus: document.getElementById('connection-status'),
    modelInfo: document.getElementById('model-info'),
    
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    realtimeTab: document.getElementById('realtime-tab'),
    batchTab: document.getElementById('batch-tab'),
    
    // Real-time settings
    chunkSize: document.getElementById('chunk-size'),
    chunkSizeValue: document.getElementById('chunk-size-value'),
    sampleRate: document.getElementById('sample-rate'),
    showTimestamps: document.getElementById('show-timestamps'),
    
    // Recording controls
    startRecording: document.getElementById('start-recording'),
    stopRecording: document.getElementById('stop-recording'),
    
    // Visualizers
    audioVisualizer: document.getElementById('audio-visualizer'),
    latencyChart: document.getElementById('latency-chart'),
    
    // Stats
    statChunksSent: document.getElementById('stat-chunks-sent'),
    statResponses: document.getElementById('stat-responses'),
    statAvgLatency: document.getElementById('stat-avg-latency'),
    statMinLatency: document.getElementById('stat-min-latency'),
    statMaxLatency: document.getElementById('stat-max-latency'),
    statRtf: document.getElementById('stat-rtf'),
    statRecordingTime: document.getElementById('stat-recording-time'),
    statProcessingTime: document.getElementById('stat-processing-time'),
    
    // Transcription
    realtimeTranscription: document.getElementById('realtime-transcription'),
    
    // Batch
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    browseBtn: document.getElementById('browse-btn'),
    fileInfo: document.getElementById('file-info'),
    fileName: document.getElementById('file-name'),
    fileDuration: document.getElementById('file-duration'),
    fileSize: document.getElementById('file-size'),
    audioPreview: document.getElementById('audio-preview'),
    batchSettings: document.getElementById('batch-settings'),
    chunkSettings: document.getElementById('chunk-settings'),
    batchChunkSize: document.getElementById('batch-chunk-size'),
    batchChunkSizeValue: document.getElementById('batch-chunk-size-value'),
    overlapSize: document.getElementById('overlap-size'),
    overlapSizeValue: document.getElementById('overlap-size-value'),
    parallelChunks: document.getElementById('parallel-chunks'),
    startBatch: document.getElementById('start-batch'),
    batchProgress: document.getElementById('batch-progress'),
    progressBar: document.getElementById('progress-bar'),
    progressText: document.getElementById('progress-text'),
    batchChunksDone: document.getElementById('batch-chunks-done'),
    batchTimePerChunk: document.getElementById('batch-time-per-chunk'),
    batchRtf: document.getElementById('batch-rtf'),
    batchEta: document.getElementById('batch-eta'),
    batchTotalTime: document.getElementById('batch-total-time'),
    chunksList: document.getElementById('chunks-list'),
    batchOutput: document.getElementById('batch-output'),
    batchTranscription: document.getElementById('batch-transcription'),
    copyResult: document.getElementById('copy-result'),
    downloadResult: document.getElementById('download-result'),
    downloadSrt: document.getElementById('download-srt'),
    
    // Logs
    logs: document.getElementById('logs'),
    clearLogs: document.getElementById('clear-logs'),
    autoScroll: document.getElementById('auto-scroll')
};


// ============================================
// Visualizers
// ============================================

const visualizer = new AudioVisualizer(elements.audioVisualizer);
const latencyChart = new LatencyChart(elements.latencyChart);


// ============================================
// Logging
// ============================================

function log(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.innerHTML = `<span class="log-time">${time}</span>${message}`;
    elements.logs.appendChild(entry);
    
    if (elements.autoScroll.checked) {
        elements.logs.scrollTop = elements.logs.scrollHeight;
    }
    
    console.log(`[${type.toUpperCase()}] ${message}`);
}


// ============================================
// RunPod API
// ============================================

const RUNPOD_API_BASE = 'https://api.runpod.ai/v2';

async function launchRunpodWorker() {
    const apiKey = elements.runpodApiKey.value.trim();
    const endpointId = elements.runpodEndpointId.value.trim();
    
    if (!apiKey) {
        log('Введите RunPod API Key', 'error');
        return;
    }
    if (!endpointId) {
        log('Введите Endpoint ID', 'error');
        return;
    }
    
    // Сохраняем в localStorage
    localStorage.setItem('runpod_api_key', apiKey);
    localStorage.setItem('runpod_endpoint_id', endpointId);
    
    log('Запуск воркера на RunPod...', 'info');
    updateWorkerStatus('starting', 'Запуск воркера...');
    
    elements.launchWorkerBtn.disabled = true;
    
    try {
        // Отправляем запрос на запуск воркера в режиме websocket
        const response = await fetch(`${RUNPOD_API_BASE}/${endpointId}/run`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                input: {
                    action: 'websocket'
                }
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const data = await response.json();
        state.runpodJobId = data.id;
        
        log(`Job создан: ${data.id}`, 'success');
        log('Ожидание запуска воркера (cold start ~15-60 сек)...', 'info');
        updateWorkerStatus('starting', `Job: ${data.id} - Ожидание запуска...`);
        
        // Начинаем polling статуса
        startRunpodPolling(apiKey, endpointId, data.id);
        
    } catch (error) {
        log(`Ошибка запуска воркера: ${error.message}`, 'error');
        updateWorkerStatus('error', `Ошибка: ${error.message}`);
        elements.launchWorkerBtn.disabled = false;
    }
}

function startRunpodPolling(apiKey, endpointId, jobId) {
    let pollCount = 0;
    const maxPolls = 180; // 3 минуты максимум (cold start может быть долгим)
    let wsUrlFound = false;
    
    state.runpodPollingInterval = setInterval(async () => {
        pollCount++;
        
        if (pollCount > maxPolls) {
            clearInterval(state.runpodPollingInterval);
            log('Таймаут ожидания воркера (3 мин)', 'error');
            updateWorkerStatus('error', 'Таймаут');
            elements.launchWorkerBtn.disabled = false;
            return;
        }
        
        try {
            const response = await fetch(`${RUNPOD_API_BASE}/${endpointId}/status/${jobId}`, {
                headers: {
                    'Authorization': `Bearer ${apiKey}`
                }
            });
            
            const data = await response.json();
            
            // Показываем статус только каждые 5 секунд или при смене
            if (pollCount % 5 === 0) {
                log(`Статус: ${data.status} (${pollCount}с)`, 'info');
            }
            
            // Пробуем найти websocket_url в разных местах
            let wsUrl = null;
            
            // 1. В прямом output
            if (data.output && data.output.websocket_url) {
                wsUrl = data.output.websocket_url;
            }
            // 2. В stream массиве
            else if (data.stream && Array.isArray(data.stream)) {
                for (const item of data.stream) {
                    if (item && item.output && item.output.websocket_url) {
                        wsUrl = item.output.websocket_url;
                        break;
                    }
                    // Иногда stream содержит напрямую объект
                    if (item && item.websocket_url) {
                        wsUrl = item.websocket_url;
                        break;
                    }
                }
            }
            // 3. Пробуем парсить output как строку (progress_update иногда возвращает строку)
            else if (data.output && typeof data.output === 'string') {
                const match = data.output.match(/ws:\/\/[\d.]+:\d+/);
                if (match) {
                    wsUrl = match[0];
                }
            }
            
            if (wsUrl && !wsUrlFound) {
                wsUrlFound = true;
                clearInterval(state.runpodPollingInterval);
                handleWorkerReady({ websocket_url: wsUrl });
                return;
            }
            
            if (data.status === 'IN_PROGRESS') {
                updateWorkerStatus('starting', `Воркер запускается... (${pollCount}с)`);
            } else if (data.status === 'IN_QUEUE') {
                updateWorkerStatus('starting', `В очереди... (${pollCount}с)`);
            } else if (data.status === 'COMPLETED') {
                clearInterval(state.runpodPollingInterval);
                if (wsUrl || (data.output && data.output.websocket_url)) {
                    handleWorkerReady(data.output);
                } else {
                    log('Воркер завершился. Output: ' + JSON.stringify(data.output), 'warning');
                    updateWorkerStatus('error', 'Воркер завершился');
                    elements.launchWorkerBtn.disabled = false;
                }
            } else if (data.status === 'FAILED') {
                clearInterval(state.runpodPollingInterval);
                log(`Воркер упал: ${JSON.stringify(data.error || data.output)}`, 'error');
                updateWorkerStatus('error', 'Ошибка воркера');
                elements.launchWorkerBtn.disabled = false;
            }
            
        } catch (error) {
            log(`Ошибка polling: ${error.message}`, 'warning');
        }
        
    }, 1000); // Каждую секунду
}

function handleWorkerReady(output) {
    const wsUrl = output.websocket_url;
    
    log(`Воркер готов! WebSocket: ${wsUrl}`, 'success');
    updateWorkerStatus('running', `Воркер запущен: ${wsUrl}`);
    
    state.runpodWorkerRunning = true;
    elements.stopWorkerBtn.disabled = false;
    
    // Автоматически подключаемся
    elements.wsUrl.value = wsUrl;
    connect();
}

async function stopRunpodWorker() {
    if (state.runpodPollingInterval) {
        clearInterval(state.runpodPollingInterval);
    }
    
    // Отключаемся от WebSocket
    if (state.websocket) {
        // Отправляем команду shutdown
        sendCommand({ action: 'shutdown' });
        setTimeout(() => {
            disconnect();
        }, 500);
    }
    
    // Можно также принудительно отменить job через API
    const apiKey = elements.runpodApiKey.value.trim();
    const endpointId = elements.runpodEndpointId.value.trim();
    
    if (apiKey && endpointId && state.runpodJobId) {
        try {
            await fetch(`${RUNPOD_API_BASE}/${endpointId}/cancel/${state.runpodJobId}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`
                }
            });
            log('Job отменён', 'info');
        } catch (e) {
            // Игнорируем ошибки отмены
        }
    }
    
    state.runpodWorkerRunning = false;
    state.runpodJobId = null;
    
    updateWorkerStatus('', 'Воркер остановлен');
    elements.launchWorkerBtn.disabled = false;
    elements.stopWorkerBtn.disabled = true;
    
    log('Воркер остановлен', 'info');
}

function updateWorkerStatus(status, text) {
    elements.workerStatus.className = `worker-status ${status}`;
    
    let html = '';
    if (status === 'starting') {
        html = '<div class="spinner"></div>';
    }
    html += `<span class="status-text">${text}</span>`;
    
    elements.workerStatus.innerHTML = html;
}

// Загрузка сохранённых настроек
function loadSavedSettings() {
    const savedApiKey = localStorage.getItem('runpod_api_key');
    const savedEndpointId = localStorage.getItem('runpod_endpoint_id');
    
    if (savedApiKey) {
        elements.runpodApiKey.value = savedApiKey;
    }
    if (savedEndpointId) {
        elements.runpodEndpointId.value = savedEndpointId;
    }
}


// ============================================
// WebSocket Connection
// ============================================

function connect() {
    const url = elements.wsUrl.value.trim();
    if (!url) {
        log('Введите WebSocket URL', 'error');
        return;
    }
    
    log(`Подключение к ${url}...`, 'info');
    updateConnectionStatus('connecting');
    
    try {
        state.websocket = new WebSocket(url);
        
        state.websocket.onopen = () => {
            log('Подключено!', 'success');
            state.connected = true;
            updateConnectionStatus('connected');
            updateControlsState();
        };
        
        state.websocket.onmessage = (event) => {
            handleWebSocketMessage(JSON.parse(event.data));
        };
        
        state.websocket.onerror = (error) => {
            log(`Ошибка WebSocket: ${error.message || 'Unknown error'}`, 'error');
        };
        
        state.websocket.onclose = (event) => {
            log(`Соединение закрыто: ${event.code} ${event.reason}`, 'warning');
            state.connected = false;
            state.websocket = null;
            updateConnectionStatus('disconnected');
            updateControlsState();
            
            if (state.recording) {
                stopRecording();
            }
        };
        
    } catch (error) {
        log(`Ошибка подключения: ${error.message}`, 'error');
        updateConnectionStatus('disconnected');
    }
}

function disconnect() {
    if (state.websocket) {
        state.websocket.close();
    }
}

function updateConnectionStatus(status) {
    elements.connectionStatus.className = `status status-${status}`;
    const statusText = {
        'disconnected': 'Не подключено',
        'connecting': 'Подключение...',
        'connected': 'Подключено'
    };
    elements.connectionStatus.querySelector('.status-text').textContent = statusText[status];
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'welcome':
            log(`Сервер: ${data.message}`, 'info');
            // Запрашиваем информацию о модели
            sendCommand({ action: 'info' });
            break;
            
        case 'info':
            displayModelInfo(data.model_info);
            break;
            
        case 'started':
            log(`Streaming начат (sample_rate: ${data.sample_rate}, chunk: ${data.chunk_duration}s)`, 'success');
            break;
            
        case 'transcription':
            handleTranscription(data);
            break;
            
        case 'stopped':
            log(`Streaming остановлен. Итог: ${data.final_text}`, 'info');
            break;
            
        case 'error':
            log(`Ошибка сервера: ${data.message}`, 'error');
            break;
            
        case 'pong':
            log('Healthcheck OK', 'success');
            break;
            
        default:
            log(`Неизвестное сообщение: ${JSON.stringify(data)}`, 'warning');
    }
}

function sendCommand(command) {
    if (state.websocket && state.connected) {
        state.websocket.send(JSON.stringify(command));
    }
}

function displayModelInfo(info) {
    if (!info) return;
    
    elements.modelInfo.classList.remove('hidden');
    elements.modelInfo.innerHTML = `
        <strong>Модель:</strong> ${info.model_name || 'N/A'}<br>
        <strong>Устройство:</strong> ${info.device || 'N/A'}<br>
        <strong>Тип данных:</strong> ${info.dtype || 'N/A'}<br>
        ${info.gpu ? `<strong>GPU:</strong> ${info.gpu.name} (${info.gpu.memory_total_gb?.toFixed(1)}GB)` : ''}
    `;
}


// ============================================
// Real-time Recording
// ============================================

async function startRecording() {
    if (state.recording) return;
    
    const chunkMs = parseInt(elements.chunkSize.value);
    const sampleRate = parseInt(elements.sampleRate.value);
    
    // Сбрасываем статистику
    resetStats();
    latencyChart.clear();
    elements.realtimeTranscription.textContent = '';
    
    // Создаём recorder
    state.recorder = new MicrophoneRecorder({
        sampleRate: sampleRate,
        chunkSizeMs: chunkMs,
        onChunk: handleAudioChunk,
        onVisualizerData: (data) => visualizer.draw(data)
    });
    
    try {
        // Отправляем команду start
        sendCommand({
            action: 'start',
            sample_rate: sampleRate,
            chunk_duration: chunkMs / 1000,
            timestamps: elements.showTimestamps.checked
        });
        
        await state.recorder.start();
        
        state.recording = true;
        state.recordingStartTime = Date.now();
        
        log(`Запись начата (chunk: ${chunkMs}ms, rate: ${sampleRate}Hz)`, 'success');
        updateControlsState();
        
        // Запускаем таймер записи
        startRecordingTimer();
        
    } catch (error) {
        log(`Ошибка записи: ${error.message}`, 'error');
    }
}

function stopRecording() {
    if (!state.recording) return;
    
    state.recording = false;
    
    if (state.recorder) {
        state.recorder.stop();
        state.recorder = null;
    }
    
    // Отправляем команду stop
    sendCommand({ action: 'stop' });
    
    visualizer.clear();
    
    log('Запись остановлена', 'info');
    updateControlsState();
}

function handleAudioChunk(audioBuffer, duration) {
    if (!state.connected || !state.recording) return;
    
    const chunkId = Date.now();
    
    // Сохраняем время отправки для измерения latency
    state.pendingChunks.set(chunkId, {
        sentAt: Date.now(),
        duration: duration
    });
    
    // Отправляем аудио
    state.websocket.send(audioBuffer);
    
    // Обновляем статистику
    state.stats.chunksSent++;
    state.stats.totalAudioDuration += duration;
    elements.statChunksSent.textContent = state.stats.chunksSent;
}

function handleTranscription(data) {
    const receivedAt = Date.now();
    
    // Ищем соответствующий чанк (берём самый старый pending)
    let latency = 0;
    if (state.pendingChunks.size > 0) {
        const [chunkId, chunkData] = state.pendingChunks.entries().next().value;
        latency = receivedAt - chunkData.sentAt;
        state.pendingChunks.delete(chunkId);
    }
    
    // Обновляем статистику
    state.stats.responsesReceived++;
    state.stats.latencies.push(latency);
    state.stats.totalProcessingTime += data.processing_time_ms || 0;
    
    updateLatencyStats();
    latencyChart.addPoint(latency);
    
    // Обновляем транскрипцию
    if (data.text) {
        appendTranscription(data.text, data.word_timestamps);
    }
    
    elements.statResponses.textContent = state.stats.responsesReceived;
    
    // RTF
    if (state.stats.totalAudioDuration > 0) {
        const rtf = state.stats.totalProcessingTime / 1000 / state.stats.totalAudioDuration;
        elements.statRtf.textContent = rtf.toFixed(3) + 'x';
    }
    
    elements.statProcessingTime.textContent = state.stats.totalProcessingTime.toFixed(0) + ' мс';
}

function appendTranscription(text, wordTimestamps) {
    const span = document.createElement('span');
    span.className = 'new-chunk';
    
    if (wordTimestamps && wordTimestamps.length > 0) {
        // С timestamps - делаем интерактивные слова
        wordTimestamps.forEach((word, i) => {
            const wordSpan = document.createElement('span');
            wordSpan.className = 'word';
            wordSpan.textContent = word.word || word.text || '';
            wordSpan.title = `${word.start?.toFixed(2)}s - ${word.end?.toFixed(2)}s`;
            span.appendChild(wordSpan);
            if (i < wordTimestamps.length - 1) {
                span.appendChild(document.createTextNode(' '));
            }
        });
    } else {
        span.textContent = text;
    }
    
    span.appendChild(document.createTextNode(' '));
    elements.realtimeTranscription.appendChild(span);
    
    // Убираем класс анимации через время
    setTimeout(() => span.classList.remove('new-chunk'), 1000);
    
    // Прокрутка
    elements.realtimeTranscription.scrollTop = elements.realtimeTranscription.scrollHeight;
}

function updateLatencyStats() {
    const latencies = state.stats.latencies;
    if (latencies.length === 0) return;
    
    const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const min = Math.min(...latencies);
    const max = Math.max(...latencies);
    
    elements.statAvgLatency.textContent = Math.round(avg) + ' мс';
    elements.statMinLatency.textContent = min + ' мс';
    elements.statMaxLatency.textContent = max + ' мс';
}

function startRecordingTimer() {
    const update = () => {
        if (!state.recording) return;
        
        const elapsed = (Date.now() - state.recordingStartTime) / 1000;
        elements.statRecordingTime.textContent = AudioUtils.formatTime(elapsed);
        
        requestAnimationFrame(update);
    };
    update();
}

function resetStats() {
    state.stats = {
        chunksSent: 0,
        responsesReceived: 0,
        latencies: [],
        totalAudioDuration: 0,
        totalProcessingTime: 0
    };
    state.pendingChunks.clear();
    
    elements.statChunksSent.textContent = '0';
    elements.statResponses.textContent = '0';
    elements.statAvgLatency.textContent = '- мс';
    elements.statMinLatency.textContent = '- мс';
    elements.statMaxLatency.textContent = '- мс';
    elements.statRtf.textContent = '-';
    elements.statRecordingTime.textContent = '0:00';
    elements.statProcessingTime.textContent = '0 мс';
}


// ============================================
// Batch Processing
// ============================================

async function handleFileSelect(file) {
    if (!file || !file.type.startsWith('audio/')) {
        log('Выберите аудио файл', 'error');
        return;
    }
    
    state.currentFile = file;
    
    // Показываем информацию о файле
    elements.fileInfo.classList.remove('hidden');
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = AudioUtils.formatFileSize(file.size);
    
    // Аудио preview
    const url = URL.createObjectURL(file);
    elements.audioPreview.src = url;
    
    // Получаем длительность
    elements.audioPreview.onloadedmetadata = () => {
        const duration = elements.audioPreview.duration;
        elements.fileDuration.textContent = AudioUtils.formatDuration(duration);
        
        elements.batchSettings.classList.remove('hidden');
        elements.startBatch.disabled = !state.connected;
        
        log(`Файл загружен: ${file.name} (${AudioUtils.formatDuration(duration)})`, 'success');
    };
}

async function startBatchProcessing() {
    if (!state.currentFile || !state.connected) return;
    
    const processMode = document.querySelector('input[name="process-mode"]:checked').value;
    
    state.batchProcessing = true;
    elements.startBatch.disabled = true;
    elements.batchProgress.classList.remove('hidden');
    elements.batchOutput.classList.add('hidden');
    elements.chunksList.innerHTML = '';
    
    const startTime = Date.now();
    let results = [];
    
    try {
        if (processMode === 'whole') {
            // Отправляем целиком
            results = await processBatchWhole();
        } else {
            // Разбиваем на чанки
            results = await processBatchChunked();
        }
        
        // Показываем результат
        displayBatchResults(results, startTime);
        
    } catch (error) {
        log(`Ошибка batch обработки: ${error.message}`, 'error');
    } finally {
        state.batchProcessing = false;
        elements.startBatch.disabled = false;
    }
}

async function processBatchWhole() {
    log('Обработка файла целиком...', 'info');
    
    updateBatchProgress(0, 1, 0);
    addChunkToList(0, 0, elements.audioPreview.duration, 'processing');
    
    // Загружаем FFmpeg если нужно
    if (!state.ffmpeg) {
        state.ffmpeg = new FFmpegProcessor();
        log('Загрузка FFmpeg WASM...', 'info');
        await state.ffmpeg.load((progress) => {
            log(`FFmpeg загружен: ${progress}%`, 'info');
        });
    }
    
    // Конвертируем в WAV
    log('Конвертация в WAV 16kHz...', 'info');
    const wavBlob = await state.ffmpeg.convertToWav(state.currentFile);
    
    // Отправляем через WebSocket
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
        // Временный handler для получения результата
        const originalHandler = state.websocket.onmessage;
        
        state.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'transcription' || data.type === 'stopped') {
                const processingTime = Date.now() - startTime;
                
                updateChunkStatus(0, 'completed', processingTime);
                updateBatchProgress(1, 1, processingTime);
                
                state.websocket.onmessage = originalHandler;
                
                resolve([{
                    index: 0,
                    text: data.text || data.final_text || '',
                    startTime: 0,
                    endTime: elements.audioPreview.duration,
                    processingTime: processingTime
                }]);
            } else if (data.type === 'error') {
                updateChunkStatus(0, 'error');
                state.websocket.onmessage = originalHandler;
                reject(new Error(data.message));
            } else {
                // Передаём другие сообщения в оригинальный handler
                handleWebSocketMessage(data);
            }
        };
        
        // Начинаем streaming
        sendCommand({
            action: 'start',
            sample_rate: 16000,
            timestamps: true
        });
        
        // Отправляем данные
        wavBlob.arrayBuffer().then(buffer => {
            // Конвертируем в Int16
            const view = new DataView(buffer);
            // Пропускаем WAV header (44 bytes)
            const samples = new Int16Array(buffer, 44);
            state.websocket.send(samples.buffer);
            
            // Останавливаем
            setTimeout(() => sendCommand({ action: 'stop' }), 500);
        });
    });
}

async function processBatchChunked() {
    const chunkDuration = parseInt(elements.batchChunkSize.value);
    const overlap = parseFloat(elements.overlapSize.value);
    const parallel = parseInt(elements.parallelChunks.value);
    
    log(`Разбиение на чанки по ${chunkDuration}с с перекрытием ${overlap}с...`, 'info');
    
    // Загружаем FFmpeg
    if (!state.ffmpeg) {
        state.ffmpeg = new FFmpegProcessor();
        log('Загрузка FFmpeg WASM...', 'info');
        await state.ffmpeg.load((progress) => {
            log(`FFmpeg загружен: ${progress}%`, 'info');
        });
    }
    
    // Разбиваем на чанки
    const { chunks, totalDuration } = await state.ffmpeg.splitIntoChunks(
        state.currentFile,
        chunkDuration,
        overlap,
        (progress, chunkIdx) => {
            log(`Разбиение: ${progress}% (чанк ${chunkIdx + 1})`, 'info');
        }
    );
    
    log(`Создано ${chunks.length} чанков`, 'success');
    
    // Добавляем чанки в UI
    chunks.forEach(chunk => {
        addChunkToList(chunk.index, chunk.startTime, chunk.endTime, 'pending');
    });
    
    state.batchChunks = chunks;
    
    // Обрабатываем чанки
    const results = [];
    const batchStartTime = Date.now();
    let completedChunks = 0;
    const chunkTimes = [];
    
    // Обработка по одному (для простоты с WebSocket)
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        updateChunkStatus(chunk.index, 'processing');
        
        const chunkStartTime = Date.now();
        
        try {
            const result = await processChunk(chunk);
            const chunkTime = Date.now() - chunkStartTime;
            
            results.push({
                ...result,
                processingTime: chunkTime
            });
            
            chunkTimes.push(chunkTime);
            completedChunks++;
            
            updateChunkStatus(chunk.index, 'completed', chunkTime);
            updateBatchProgress(completedChunks, chunks.length, Date.now() - batchStartTime, chunkTimes, chunks);
            
        } catch (error) {
            log(`Ошибка чанка ${chunk.index}: ${error.message}`, 'error');
            updateChunkStatus(chunk.index, 'error');
        }
    }
    
    return results;
}

function processChunk(chunk) {
    return new Promise((resolve, reject) => {
        const originalHandler = state.websocket.onmessage;
        const startTime = Date.now();
        
        state.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'transcription' || data.type === 'stopped') {
                state.websocket.onmessage = originalHandler;
                
                resolve({
                    index: chunk.index,
                    text: data.text || data.final_text || '',
                    startTime: chunk.startTime,
                    endTime: chunk.endTime,
                    wordTimestamps: data.word_timestamps || []
                });
            } else if (data.type === 'error') {
                state.websocket.onmessage = originalHandler;
                reject(new Error(data.message));
            } else if (data.type !== 'started') {
                handleWebSocketMessage(data);
            }
        };
        
        // Начинаем streaming
        sendCommand({
            action: 'start',
            sample_rate: 16000,
            timestamps: true
        });
        
        // Отправляем данные чанка
        chunk.data.arrayBuffer().then(buffer => {
            // Пропускаем WAV header
            const samples = new Int16Array(buffer, 44);
            state.websocket.send(samples.buffer);
            
            // Останавливаем
            setTimeout(() => sendCommand({ action: 'stop' }), 300);
        });
    });
}

function addChunkToList(index, startTime, endTime, status) {
    const item = document.createElement('div');
    item.className = `chunk-item ${status}`;
    item.id = `chunk-${index}`;
    item.innerHTML = `
        <span class="chunk-index">Чанк ${index + 1}</span>
        <span class="chunk-time">${AudioUtils.formatTime(startTime)} - ${AudioUtils.formatTime(endTime)}</span>
        <span class="chunk-status" id="chunk-status-${index}">${getStatusText(status)}</span>
    `;
    elements.chunksList.appendChild(item);
}

function updateChunkStatus(index, status, processingTime = null) {
    const item = document.getElementById(`chunk-${index}`);
    const statusEl = document.getElementById(`chunk-status-${index}`);
    
    if (item) {
        item.className = `chunk-item ${status}`;
    }
    if (statusEl) {
        let text = getStatusText(status);
        if (processingTime) {
            text += ` (${(processingTime / 1000).toFixed(1)}с)`;
        }
        statusEl.textContent = text;
    }
}

function getStatusText(status) {
    const texts = {
        'pending': '⏳ Ожидание',
        'processing': '🔄 Обработка...',
        'completed': '✅ Готово',
        'error': '❌ Ошибка'
    };
    return texts[status] || status;
}

function updateBatchProgress(completed, total, elapsedTime, chunkTimes = [], allChunks = []) {
    const percent = Math.round((completed / total) * 100);
    elements.progressBar.style.width = percent + '%';
    elements.progressText.textContent = percent + '%';
    elements.batchChunksDone.textContent = `${completed} / ${total}`;
    elements.batchTotalTime.textContent = (elapsedTime / 1000).toFixed(1) + ' сек';
    
    if (chunkTimes.length > 0) {
        const avgTime = chunkTimes.reduce((a, b) => a + b, 0) / chunkTimes.length;
        elements.batchTimePerChunk.textContent = (avgTime / 1000).toFixed(2) + ' сек';
        
        // RTF
        const totalAudioProcessed = allChunks.slice(0, completed).reduce((sum, c) => sum + c.duration, 0);
        const rtf = (elapsedTime / 1000) / totalAudioProcessed;
        elements.batchRtf.textContent = rtf.toFixed(2) + 'x';
        
        // ETA
        const remaining = total - completed;
        const eta = (avgTime * remaining) / 1000;
        elements.batchEta.textContent = AudioUtils.formatDuration(eta);
    }
}

function displayBatchResults(results, startTime) {
    const totalTime = (Date.now() - startTime) / 1000;
    
    elements.batchOutput.classList.remove('hidden');
    
    // Сортируем по времени начала
    results.sort((a, b) => a.startTime - b.startTime);
    
    // Собираем полный текст
    const fullText = results.map(r => r.text).join(' ').trim();
    elements.batchTranscription.textContent = fullText;
    
    // Сохраняем для экспорта
    state.batchResults = results;
    
    log(`Обработка завершена за ${totalTime.toFixed(1)}с`, 'success');
}


// ============================================
// Export Functions
// ============================================

function copyResults() {
    const text = elements.batchTranscription.textContent;
    navigator.clipboard.writeText(text).then(() => {
        log('Текст скопирован в буфер обмена', 'success');
    });
}

function downloadTxt() {
    const text = elements.batchTranscription.textContent;
    const blob = new Blob([text], { type: 'text/plain' });
    downloadBlob(blob, 'transcription.txt');
}

function downloadSrt() {
    if (!state.batchResults) return;
    
    let srt = '';
    let index = 1;
    
    state.batchResults.forEach(result => {
        if (result.wordTimestamps && result.wordTimestamps.length > 0) {
            // Группируем слова в сегменты по ~5 секунд
            let segmentWords = [];
            let segmentStart = result.startTime;
            
            result.wordTimestamps.forEach((word, i) => {
                segmentWords.push(word.word || word.text || '');
                
                const wordEnd = result.startTime + (word.end || 0);
                const shouldBreak = (wordEnd - segmentStart > 5) || (i === result.wordTimestamps.length - 1);
                
                if (shouldBreak && segmentWords.length > 0) {
                    const segmentEnd = wordEnd;
                    srt += `${index}\n`;
                    srt += `${formatSrtTime(segmentStart)} --> ${formatSrtTime(segmentEnd)}\n`;
                    srt += `${segmentWords.join(' ')}\n\n`;
                    
                    index++;
                    segmentWords = [];
                    segmentStart = segmentEnd;
                }
            });
        } else {
            // Без timestamps - один сегмент на чанк
            srt += `${index}\n`;
            srt += `${formatSrtTime(result.startTime)} --> ${formatSrtTime(result.endTime)}\n`;
            srt += `${result.text}\n\n`;
            index++;
        }
    });
    
    const blob = new Blob([srt], { type: 'text/plain' });
    downloadBlob(blob, 'transcription.srt');
}

function formatSrtTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}


// ============================================
// UI Helpers
// ============================================

function updateControlsState() {
    const connected = state.connected;
    const recording = state.recording;
    
    elements.connectBtn.disabled = connected;
    elements.disconnectBtn.disabled = !connected;
    elements.healthcheckBtn.disabled = !connected;
    
    elements.startRecording.disabled = !connected || recording;
    elements.stopRecording.disabled = !recording;
    
    elements.startBatch.disabled = !connected || !state.currentFile || state.batchProcessing;
}


// ============================================
// Event Listeners
// ============================================

// RunPod
elements.launchWorkerBtn.addEventListener('click', launchRunpodWorker);
elements.stopWorkerBtn.addEventListener('click', stopRunpodWorker);

// Connection (direct WebSocket)
elements.connectBtn.addEventListener('click', connect);
elements.disconnectBtn.addEventListener('click', disconnect);
elements.healthcheckBtn.addEventListener('click', () => sendCommand({ action: 'ping' }));

// Tabs
elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        elements.tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.getElementById(`${tab}-tab`).classList.add('active');
    });
});

// Real-time settings
elements.chunkSize.addEventListener('input', () => {
    const value = elements.chunkSize.value;
    elements.chunkSizeValue.textContent = value + ' мс';
    if (state.recorder) {
        state.recorder.setChunkSize(parseInt(value));
    }
});

// Recording
elements.startRecording.addEventListener('click', startRecording);
elements.stopRecording.addEventListener('click', stopRecording);

// File upload
elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag & drop
elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('dragover');
});

elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('dragover');
});

elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

// Batch settings
document.querySelectorAll('input[name="process-mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        elements.chunkSettings.classList.toggle('hidden', e.target.value === 'whole');
    });
});

elements.batchChunkSize.addEventListener('input', () => {
    elements.batchChunkSizeValue.textContent = elements.batchChunkSize.value + ' сек';
});

elements.overlapSize.addEventListener('input', () => {
    elements.overlapSizeValue.textContent = elements.overlapSize.value + ' сек';
});

elements.startBatch.addEventListener('click', startBatchProcessing);

// Export
elements.copyResult.addEventListener('click', copyResults);
elements.downloadResult.addEventListener('click', downloadTxt);
elements.downloadSrt.addEventListener('click', downloadSrt);

// Logs
elements.clearLogs.addEventListener('click', () => {
    elements.logs.innerHTML = '';
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Space to start/stop recording (when not in input)
    if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        if (state.recording) {
            stopRecording();
        } else if (state.connected) {
            startRecording();
        }
    }
});


// ============================================
// Initialization
// ============================================

// Загружаем сохранённые настройки
loadSavedSettings();

log('Parakeet ASR Frontend загружен', 'success');
log('Нажмите "Запустить воркер" для начала работы', 'info');

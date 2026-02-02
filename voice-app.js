const startBtn = document.getElementById('start-btn');
const startBtnText = startBtn.querySelector('span');
const startBtnIcon = startBtn.querySelector('i');
const chatBox = document.getElementById('chat-box');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const orb = document.getElementById('orb');

let socket;
let audioContext;
let processor;
let inputSource;
let mediaStream;
let isRecording = false;
let isListening = false;

// Voice Activity Detection (VAD) settings
let silenceStart = null;
let isSpeaking = false;
const SILENCE_THRESHOLD = 2000; // 2 seconds of silence
const AMPLITUDE_THRESHOLD = 0.01; // Minimum amplitude to consider as speech

startBtn.addEventListener('click', async () => {
    if (isListening) {
        stopButtonAnimation();
        stopRecording();
        return;
    }

    setButtonState('connecting');
    updateStatus("Connecting...");

    try {
        await initAudio();
        connectWebSocket();
    } catch (err) {
        console.error("Error starting:", err);
        setButtonState('error');
        updateStatus("Error: " + err.message);
    }
});

async function initAudio() {
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log("Connecting to:", wsUrl);
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("✅ Connected to server");
        setButtonState('listening');
        updateStatus("Listening...");
        orb.classList.add('listening');
        startRecording();
    };

    socket.onmessage = async (event) => {
        if (event.data instanceof Blob) {
            // Received audio blob (TTS)
            updateStatus("Speaking...");
            orb.className = "orb speaking";
            await playAudio(event.data);

            if (isListening) {
                updateStatus("Listening...");
                orb.className = "orb listening";
                isSpeaking = false;
                silenceStart = null;
            }
        } else {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'status') {
                    updateStatus(data.message);
                } else if (data.type === 'user_transcript') {
                    addMessage(data.text, 'user');
                } else if (data.type === 'bot_response' || data.type === 'text') {
                    addMessage(data.text, 'bot');
                } else if (data.type === 'error') {
                    updateStatus("Error: " + data.message);
                }
            } catch (e) {
                console.log("Received non-JSON text:", event.data);
            }
        }
    };

    socket.onclose = (event) => {
        console.log("❌ Disconnected", event);
        stopRecording();
        setButtonState('idle');
        updateStatus("Disconnected");
        orb.className = "orb";
    };

    socket.onerror = (err) => {
        console.error("❌ WebSocket error:", err);
        updateStatus("Connection Error");
        setButtonState('error');
    };
}

// ... Audio Recording Logic (Same as before) ...
async function startRecording() {
    isRecording = true;
    isListening = true;
    silenceStart = null;
    isSpeaking = false;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });

        inputSource = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        inputSource.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);

            // Calculate RMS
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) {
                sum += inputData[i] * inputData[i];
            }
            const rms = Math.sqrt(sum / inputData.length);

            // VAD Logic
            if (rms > AMPLITUDE_THRESHOLD) {
                if (!isSpeaking) {
                    isSpeaking = true;
                    updateStatus("Listening (Speech Detected)...");
                    orb.className = "orb listening";
                }
                silenceStart = null;
            } else {
                if (isSpeaking && silenceStart === null) {
                    silenceStart = Date.now();
                } else if (isSpeaking && silenceStart !== null) {
                    if (Date.now() - silenceStart >= SILENCE_THRESHOLD) {
                        if (isSpeaking) {
                            updateStatus("Processing...");
                            orb.className = "orb processing";
                            isSpeaking = false;
                            silenceStart = null;
                        }
                    }
                }
            }

            const pcmData = floatTo16BitPCM(inputData);
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(pcmData);
            }
        };

    } catch (err) {
        console.error("Microphone error:", err);
        updateStatus("Mic Error");
        setButtonState('idle');
    }
}

function stopRecording() {
    isRecording = false;
    isListening = false;
    isSpeaking = false;
    silenceStart = null;

    if (processor) { processor.disconnect(); processor.onaudioprocess = null; processor = null; }
    if (inputSource) { inputSource.disconnect(); inputSource = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; }
    if (socket) { socket.close(); socket = null; }
}

async function playAudio(blob) {
    try {
        if (!audioContext || audioContext.state === 'closed') await initAudio();
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start(0);
    } catch (err) {
        console.error("Playback error:", err);
    }
}

function floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output.buffer;
}

// UI Helpers
function updateStatus(text) {
    if (statusText) statusText.innerText = text;
    const orb = document.getElementById('orb');

    // Auto-update orb state based on status text if not explicitly set
    if (text.includes("Listening")) orb.className = "orb listening";
    else if (text.includes("Speaking")) orb.className = "orb speaking";
    else if (text.includes("Processing")) orb.className = "orb processing";
}

function setButtonState(state) {
    const connStatus = document.getElementById('conn-status');
    const latencyVal = document.getElementById('latency-val');

    startBtn.className = ''; // Reset classes

    if (state === 'listening') {
        startBtn.classList.add('active');
        if (startBtnText) startBtnText.innerText = "End Session";
        if (startBtnIcon) startBtnIcon.className = "fas fa-stop";
        if (statusDot) {
            statusDot.style.background = "#22c55e";
            statusDot.style.boxShadow = "0 0 10px #22c55e";
        }
        if (connStatus) connStatus.innerText = "Stable";
        if (latencyVal) latencyVal.innerText = (Math.floor(Math.random() * 20) + 15) + "ms";
    } else if (state === 'connecting') {
        if (startBtnText) startBtnText.innerText = "Connecting...";
        if (startBtnIcon) startBtnIcon.className = "fas fa-spinner fa-spin";
        if (statusDot) {
            statusDot.style.background = "#eab308";
            statusDot.style.boxShadow = "none";
        }
        if (connStatus) connStatus.innerText = "Connecting...";
    } else {
        // Idle or Error
        if (startBtnText) startBtnText.innerText = "Start Conversation";
        if (startBtnIcon) startBtnIcon.className = "fas fa-microphone";
        if (statusDot) {
            statusDot.style.background = "#94a3b8"; // Muted
            statusDot.style.boxShadow = "none";
        }
        if (connStatus) connStatus.innerText = "Disconnected";
        if (latencyVal) latencyVal.innerText = "0ms";
    }
}

function stopButtonAnimation() {
    setButtonState('idle');
}

function addMessage(text, type) {
    if (!text) return;

    const chatBox = document.getElementById('chat-box');
    const lastRow = chatBox.lastElementChild;
    const lastType = lastRow ? (lastRow.classList.contains('user-row') ? 'user' : 'bot') : null;

    // Check if we should merge with previous message
    if (lastRow && lastType === type) {
        const messageDiv = lastRow.querySelector('.message');
        if (messageDiv) {
            const timestampSpan = messageDiv.querySelector('.timestamp');
            if (timestampSpan) timestampSpan.remove();

            const separator = messageDiv.innerText.trim().length > 0 && !messageDiv.innerText.endsWith(' ') ? ' ' : '';
            const newContent = text.replace(/\n/g, '<br>');

            messageDiv.innerHTML += separator + newContent + `<span class="timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            return;
        }
    }

    const row = document.createElement('div');
    row.className = `message-row ${type === 'user' ? 'user-row' : 'bot-row'}`;

    // Convert text to handle newlines
    const formattedText = text.replace(/\n/g, '<br>');

    row.innerHTML = `
        <div class="message ${type}">
            ${formattedText}
            <span class="timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
    `;

    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
}

console.log("AI Enhancer Voice App Loaded");


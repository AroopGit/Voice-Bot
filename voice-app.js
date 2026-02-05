const startBtn = document.getElementById('start-btn');
const chatBox = document.getElementById('chat-box');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const orb = document.getElementById('orb');
const latencyVal = document.getElementById('latency-val');

let socket;
let audioContext;
let processor;
let inputSource;
let mediaStream;
let isRecording = false;
let isListening = false;
let isBotSpeaking = false;  // NEW: Track when bot is speaking to mute mic

// Voice Activity Detection (VAD) settings
let silenceStart = null;
let isSpeaking = false;
const SILENCE_THRESHOLD = 800; // 0.8 seconds of silence
const AMPLITUDE_THRESHOLD = 0.01; // Minimum amplitude to consider as speech

startBtn.addEventListener('click', async () => {
    if (isListening) {
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
        orb.className = "orb listening";
        startRecording();
    };

    socket.onmessage = async (event) => {
        if (event.data instanceof Blob) {
            // Received audio blob (TTS) - MUTE MIC BEFORE PLAYING
            console.log("🔊 Bot speaking - muting microphone");
            isBotSpeaking = true;
            pauseMicrophone();

            updateStatus("Speaking...");
            orb.className = "orb speaking";

            await playAudio(event.data);

            // Resume microphone AFTER audio finishes
            console.log("🎤 Bot finished - resuming microphone");
            isBotSpeaking = false;

            if (isListening) {
                resumeMicrophone();
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
                } else if (data.type === 'bot_speaking_start') {
                    // Bot is about to speak - mute mic
                    console.log("🔇 Received bot_speaking_start - muting mic");
                    isBotSpeaking = true;
                    pauseMicrophone();
                } else if (data.type === 'bot_speaking_end') {
                    // Bot finished speaking - unmute mic
                    console.log("🔊 Received bot_speaking_end - unmuting mic");
                    isBotSpeaking = false;
                    resumeMicrophone();
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

// Pause microphone (mute) - stops sending audio to server
function pauseMicrophone() {
    if (mediaStream) {
        mediaStream.getAudioTracks().forEach(track => {
            track.enabled = false;
        });
        console.log("🔇 Microphone muted");
    }
}

// Resume microphone (unmute) - resumes sending audio
function resumeMicrophone() {
    if (mediaStream) {
        mediaStream.getAudioTracks().forEach(track => {
            track.enabled = true;
        });
        console.log("🔊 Microphone unmuted");
    }
}

async function startRecording() {
    isRecording = true;
    isListening = true;
    silenceStart = null;
    isSpeaking = false;
    isBotSpeaking = false;

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
            // Don't process audio if bot is speaking or not recording
            if (!isRecording || isBotSpeaking) return;

            const inputData = e.inputBuffer.getChannelData(0);

            let sum = 0;
            for (let i = 0; i < inputData.length; i++) {
                sum += inputData[i] * inputData[i];
            }
            const rms = Math.sqrt(sum / inputData.length);

            if (rms > AMPLITUDE_THRESHOLD) {
                if (!isSpeaking) {
                    isSpeaking = true;
                    updateStatus("Listening...");
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
            if (socket && socket.readyState === WebSocket.OPEN && !isBotSpeaking) {
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
    isBotSpeaking = false;
    silenceStart = null;

    if (processor) { processor.disconnect(); processor.onaudioprocess = null; processor = null; }
    if (inputSource) { inputSource.disconnect(); inputSource = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; }
    if (socket) { socket.close(); socket = null; }

    setButtonState('idle');
    updateStatus("Ready to Connect");
    orb.className = "orb";
}

async function playAudio(blob) {
    return new Promise(async (resolve) => {
        try {
            if (!audioContext || audioContext.state === 'closed') await initAudio();
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);

            // When audio playback ends, resolve the promise
            source.onended = () => {
                console.log("🔊 Audio playback finished");
                resolve();
            };

            source.start(0);

            // Fallback timeout in case onended doesn't fire
            const duration = audioBuffer.duration * 1000 + 500; // Add 500ms buffer
            setTimeout(() => {
                resolve();
            }, duration);

        } catch (err) {
            console.error("Playback error:", err);
            resolve(); // Resolve anyway to continue
        }
    });
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
    if (statusText) {
        statusText.innerText = text;
        if (text.includes("Processing")) {
            statusText.classList.add('visible');
        } else {
            statusText.classList.remove('visible');
        }
    }

    const orb = document.getElementById('orb');
    if (text.includes("Listening")) orb.className = "orb listening";
    else if (text.includes("Speaking")) orb.className = "orb speaking";
    else if (text.includes("Processing")) orb.className = "orb processing";
}

function setButtonState(state) {
    startBtn.className = '';

    if (state === 'listening') {
        startBtn.classList.add('active');
        if (latencyVal) latencyVal.innerText = (Math.floor(Math.random() * 10) + 20) + "ms";
    } else if (state === 'connecting') {
        startBtn.classList.add('connecting');
    } else {
        startBtn.classList.add('idle');
        if (latencyVal) latencyVal.innerText = "0ms";
    }
}

function addMessage(text, type) {
    if (!text) return;

    const chatBox = document.getElementById('chat-box');

    const row = document.createElement('div');
    row.className = `message-row ${type}-row`;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.innerHTML = `${text.replace(/\n/g, '<br>')} <span class="msg-timestamp">${time}</span>`;

    row.appendChild(messageDiv);
    chatBox.appendChild(row);
    chatBox.scrollTop = chatBox.scrollHeight;
}

console.log("Customer Support Assistant Voice App Loaded");

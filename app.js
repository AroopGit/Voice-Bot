// Multilingual Voice Conversation POC - Frontend Logic
// Uses Browser's Web Speech API for zero-cost audio handling

class VoiceAssistant {
    constructor() {
        // DOM Elements
        this.micButton = document.getElementById('micButton');
        this.micIcon = document.getElementById('micIcon');
        this.statusText = document.getElementById('statusText');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.transcriptContainer = document.getElementById('transcriptContainer');
        this.clearButton = document.getElementById('clearButton');

        // Intelligence Panel Elements
        this.languageValue = document.getElementById('languageValue');
        this.subjectValue = document.getElementById('subjectValue');
        this.concernsValue = document.getElementById('concernsValue');
        this.queriesValue = document.getElementById('queriesValue');

        // State
        this.isListening = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;

        // Initialize
        this.initSpeechRecognition();
        this.attachEventListeners();
    }

    initSpeechRecognition() {
        // Check for browser support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            this.showError('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
            return;
        }

        // Initialize recognition
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;  // Keep listening until manually stopped
        this.recognition.interimResults = true;  // Show interim results while speaking
        this.recognition.maxAlternatives = 1;

        // Track if we've received any speech
        this.lastSpeechTime = null;
        this.speechTimeout = null;

        // Event handlers
        this.recognition.onstart = () => {
            this.updateStatus('Listening... (Click again to stop)', 'listening');
        };

        this.recognition.onresult = (event) => {
            // Get the latest result
            const last = event.results.length - 1;
            const transcript = event.results[last][0].transcript;

            // Update last speech time
            this.lastSpeechTime = Date.now();

            // Clear any existing timeout
            if (this.speechTimeout) {
                clearTimeout(this.speechTimeout);
            }

            // If this is a final result, process it
            if (event.results[last].isFinal) {
                // Wait 1.5 seconds of silence before auto-stopping
                this.speechTimeout = setTimeout(() => {
                    if (this.isListening) {
                        this.handleUserSpeech(transcript);
                        this.stopListening();
                    }
                }, 1500);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);

            // Don't show error for "no-speech" - just keep listening
            if (event.error === 'no-speech') {
                this.updateStatus('No speech detected. Try again...', 'listening');
                return;
            }

            this.updateStatus(`Error: ${event.error}`, 'error');
            this.stopListening();
        };

        this.recognition.onend = () => {
            // Clear timeout if exists
            if (this.speechTimeout) {
                clearTimeout(this.speechTimeout);
            }

            if (this.isListening) {
                this.stopListening();
            }
        };
    }

    attachEventListeners() {
        this.micButton.addEventListener('click', () => this.toggleListening());
        this.clearButton.addEventListener('click', () => this.clearTranscript());
    }

    toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    startListening() {
        if (!this.recognition) {
            this.showError('Speech recognition not initialized');
            return;
        }

        try {
            this.recognition.start();
            this.isListening = true;
            this.micButton.classList.add('active');
            this.updateStatus('Listening...', 'listening');
        } catch (error) {
            console.error('Error starting recognition:', error);
            this.showError('Failed to start listening');
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            // Clear any pending speech timeout
            if (this.speechTimeout) {
                clearTimeout(this.speechTimeout);
                this.speechTimeout = null;
            }

            this.recognition.stop();
            this.isListening = false;
            this.micButton.classList.remove('active');
            this.updateStatus('Ready to listen', 'ready');
        }
    }

    async handleUserSpeech(transcript) {
        // Add user message to transcript
        this.addTranscriptMessage('user', transcript);

        try {
            // Send to backend for processing
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ transcript })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to process transcript');
            }

            const data = await response.json();

            // Update intelligence panel
            this.updateIntelligence(data);

            // Add bot reply to transcript
            this.addTranscriptMessage('bot', data.bot_reply);

            // Speak the bot's reply
            this.speak(data.bot_reply, data.language_id);

            this.updateStatus('Ready to listen', 'ready');

        } catch (error) {
            console.error('Error processing speech:', error);
            this.showError(error.message);
            this.updateStatus('Error occurred', 'error');
        }
    }

    speak(text, language) {
        // Cancel any ongoing speech
        this.synthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        // Set language based on detected language
        const langMap = {
            'English': 'en-US',
            'Spanish': 'es-ES',
            'French': 'fr-FR',
            'German': 'de-DE',
            'Italian': 'it-IT',
            'Portuguese': 'pt-PT',
            'Hindi': 'hi-IN',
            'Chinese': 'zh-CN',
            'Japanese': 'ja-JP',
            'Korean': 'ko-KR',
            'Arabic': 'ar-SA',
            'Russian': 'ru-RU'
        };

        utterance.lang = langMap[language] || 'en-US';
        utterance.rate = 0.9;
        utterance.pitch = 1;

        this.synthesis.speak(utterance);
    }

    addTranscriptMessage(type, text) {
        // Clear placeholder if exists
        const placeholder = this.transcriptContainer.querySelector('.text-slate-500');
        if (placeholder) {
            this.transcriptContainer.innerHTML = '';
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `transcript-item p-4 rounded-lg ${type === 'user'
            ? 'bg-blue-900/30 border-l-4 border-blue-500'
            : 'bg-purple-900/30 border-l-4 border-purple-500'
            }`;

        messageDiv.innerHTML = `
            <div class="flex items-center mb-1">
                <span class="text-xs font-semibold ${type === 'user' ? 'text-blue-400' : 'text-purple-400'
            }">
                    ${type === 'user' ? '👤 You' : '🤖 Assistant'}
                </span>
                <span class="text-xs text-slate-500 ml-2">${new Date().toLocaleTimeString()}</span>
            </div>
            <p class="text-white">${this.escapeHtml(text)}</p>
        `;

        this.transcriptContainer.appendChild(messageDiv);
        this.transcriptContainer.scrollTop = this.transcriptContainer.scrollHeight;
    }

    updateIntelligence(data) {
        // Update language
        this.languageValue.textContent = data.language_id;

        // Update subject
        this.subjectValue.textContent = data.subject || '—';

        // Update concerns
        if (data.concerns && data.concerns.length > 0) {
            this.concernsValue.innerHTML = data.concerns
                .map(concern => `
                    <div class="flex items-start">
                        <span class="text-yellow-400 mr-2">⚠</span>
                        <span class="text-white text-sm">${this.escapeHtml(concern)}</span>
                    </div>
                `).join('');
        } else {
            this.concernsValue.innerHTML = '<p class="text-slate-500 text-sm">None detected</p>';
        }

        // Update queries
        if (data.queries && data.queries.length > 0) {
            this.queriesValue.innerHTML = data.queries
                .map(query => `
                    <div class="flex items-start">
                        <span class="text-purple-400 mr-2">❓</span>
                        <span class="text-white text-sm">${this.escapeHtml(query)}</span>
                    </div>
                `).join('');
        } else {
            this.queriesValue.innerHTML = '<p class="text-slate-500 text-sm">None detected</p>';
        }
    }

    clearTranscript() {
        this.transcriptContainer.innerHTML = `
            <div class="text-slate-500 text-center py-8">
                <svg class="w-16 h-16 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                </svg>
                <p>Start speaking to see the conversation here</p>
            </div>
        `;

        // Reset intelligence panel
        this.languageValue.textContent = '—';
        this.subjectValue.textContent = '—';
        this.concernsValue.innerHTML = '<p class="text-slate-500 text-sm">None detected</p>';
        this.queriesValue.innerHTML = '<p class="text-slate-500 text-sm">None detected</p>';
    }

    updateStatus(message, state) {
        this.statusText.textContent = message;

        // Update indicator color
        this.statusIndicator.className = 'status-indicator';
        switch (state) {
            case 'listening':
                this.statusIndicator.classList.add('bg-red-500');
                break;
            case 'processing':
                this.statusIndicator.classList.add('bg-yellow-500');
                break;
            case 'error':
                this.statusIndicator.classList.add('bg-red-600');
                break;
            default:
                this.statusIndicator.classList.add('bg-green-500');
        }
    }

    showError(message) {
        this.addTranscriptMessage('bot', `Error: ${message}`);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the voice assistant when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new VoiceAssistant();
});

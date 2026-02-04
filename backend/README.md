# SwiftShip India Voice Bot Backend 🇮🇳

High-performance RAG voice bot for logistics customer service, optimized for the **Indian market** with support for **Hindi, English, and Hinglish**.

## 🚀 Features

- **Trilingual Support**: Hindi (हिंदी), English, and Hinglish (code-mixed)
- **Indian TTS Voices**: Natural Indian English and Hindi voices
- **Real-time Voice Processing**: WebSocket-based streaming for < 3-second latency
- **RAG-powered Responses**: Context-aware answers from knowledge base
- **Indian Logistics Focus**: COD, PIN codes, festival delivery, regional coverage

## 🗣️ Language Support

| Language | Script | TTS Voice | Example Query |
|----------|--------|-----------|---------------|
| English | Latin | en-IN-NeerjaNeural | "Where is my package?" |
| Hindi | Devanagari | hi-IN-SwaraNeural | "मेरा पैकेज कहाँ है?" |
| Hinglish | Latin (Romanized) | en-IN-NeerjaNeural | "Mera package kaha hai?" |

### Hinglish Detection Examples
The bot automatically detects when customers use Hinglish:
- "Delivery kab hoga?"
- "COD available hai kya?"
- "Tracking number ka status batao"
- "Order cancel karna hai"

## 📋 Tech Stack

| Component | Technology |
|-----------|------------|
| STT | Faster-Whisper (GPU/CPU) |
| Language Detection | Rule-based (Devanagari + Hinglish keywords) |
| Embeddings | Sentence-Transformers (multilingual) |
| Vector DB | Qdrant |
| LLM | Groq (Llama 3.1) |
| TTS | Edge-TTS (Indian voices) |
| Backend | FastAPI + WebSocket |

## 🛠 Quick Start

### Prerequisites

1. **Python 3.10+** installed
2. **Qdrant** running (Docker or local)
3. **Groq API Key** from [console.groq.com](https://console.groq.com)

### Installation

```powershell
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (if using Docker)
docker run -p 6333:6333 qdrant/qdrant
```

### Configuration

```powershell
# Add your Groq API key to .env
notepad .env
# Set: GROQ_API_KEY=your_key_here
```

### Start Server

```powershell
# Run setup first (downloads models, loads knowledge base)
python setup.py

# Start the server
python main.py
```

Server available at `http://localhost:8000`

## 📡 API Endpoints

### Voice Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Real-time voice streaming |
| `/api/voice-query` | POST | Audio in → Audio out |
| `/api/text-query` | POST | Text query (for testing) |

### Knowledge Base

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload-knowledge` | POST | Upload single document |
| `/api/search` | GET | Search knowledge base |

### Utilities

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/languages` | GET | List supported languages |

## 📚 Knowledge Base Topics

The bot comes pre-loaded with Indian logistics knowledge:

- **PIN Code Serviceability**: 1xxxx to 8xxxxx zones
- **Metro City Delivery**: Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune
- **COD Policies**: Limits up to ₹50,000, UPI at doorstep
- **Festival Delivery**: Diwali, Holi, Eid, Durga Puja delays
- **Address Format**: Indian address standards, landmarks
- **RTO Policies**: Return to Origin procedures
- **State-wise Coverage**: All states and union territories

## 🎯 Sample Interactions

### English
```
User: "Where is my package? Tracking number is ABC123"
Bot: "Let me check that for you. Your package ABC123 is currently in transit 
     and will be delivered by tomorrow evening to your address in Mumbai."
```

### Hindi
```
User: "मेरा पैकेज कहाँ है? ट्रैकिंग नंबर ABC123 है"
Bot: "मैं आपके लिए चेक करती हूं। आपका पैकेज ABC123 अभी ट्रांजिट में है 
     और कल शाम तक मुंबई में आपके पते पर डिलीवर हो जाएगा।"
```

### Hinglish
```
User: "Mera package kaha hai? Tracking number ABC123 hai"
Bot: "Main check karti hoon. Aapka package ABC123 abhi transit mein hai 
     aur kal shaam tak Mumbai mein aapke address pe deliver ho jayega."
```

## 🧪 Testing

```powershell
# Run all tests including language detection
cd backend
python test_voice_bot.py
```

Test cases include:
- ✅ English query detection
- ✅ Hindi (Devanagari) query detection
- ✅ Hinglish (Romanized Hindi) detection
- ✅ Indian TTS voice synthesis
- ✅ Same-language response generation

## ⚡ Performance

| Step | Target | Typical |
|------|--------|---------|
| Language Detection | < 100ms | 10-30ms |
| STT (Whisper) | < 500ms | 200-400ms |
| RAG Retrieval | < 200ms | 50-150ms |
| LLM Generation | < 1500ms | 500-1000ms |
| TTS Synthesis | < 500ms | 200-400ms |
| **Total Pipeline** | **< 3000ms** | **1000-2000ms** |

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application
├── config.py               # Configuration (Indian system prompt)
├── setup.py                # Setup script
├── test_voice_bot.py       # Test suite
├── requirements.txt        # Dependencies
├── .env                    # Configuration (add GROQ_API_KEY)
├── services/
│   ├── stt_service.py      # Faster-Whisper STT
│   ├── tts_service.py      # Edge-TTS (Indian voices)
│   ├── rag_service.py      # Qdrant + Groq RAG
│   ├── language_detector.py # Hindi/English/Hinglish detection
│   └── streaming_service.py # Pipeline orchestration
└── data/knowledge_base/    # Indian logistics FAQs
    ├── pin_code_serviceability.txt
    ├── metro_city_delivery.txt
    ├── cod_policy.txt
    ├── festival_delivery.txt
    ├── address_format_india.txt
    ├── state_coverage.txt
    ├── tracking_system.txt
    ├── contact_support.txt
    ├── rto_policy.txt
    └── faqs.txt
```

## 🔧 Troubleshooting

### "Qdrant connection error"
```powershell
# Make sure Qdrant is running
docker run -p 6333:6333 qdrant/qdrant
```

### "Hinglish not detected correctly"
The system uses keyword-based detection. Common Hinglish words like `hai`, `kya`, `kab`, `mera` trigger Hinglish mode.

### "Hindi TTS voice not working"
Ensure you have internet connection - Edge-TTS requires API access.

## 🇮🇳 Indian-Specific Features

- **PIN Code Validation**: 6-digit format with zone detection
- **COD Support**: Up to ₹50,000 with UPI/Card options
- **Festival Calendar**: Diwali, Holi, Eid, Durga Puja considerations
- **Regional Coverage**: All 28 states + 8 UTs
- **IST Time Support**: Business hours in India Standard Time
- **Toll-Free Support**: 1800-SWIFT-IN format

## 📄 License

MIT License - see LICENSE file for details.

---

**Made for India 🇮🇳** | SwiftShip Logistics

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import json
import os
from typing import Optional
import numpy as np
import uuid
import re
from datetime import datetime

# Import services
from app.services.stt import FasterWhisperSTT
from app.services.llm import MistralLLM
from app.services.tts import EdgeTTSService
from app.services.rag import RAGService
from app.services.translation import TranslationService

# Global service instances
stt_service: Optional[FasterWhisperSTT] = None
llm_service: Optional[MistralLLM] = None
tts_service: Optional[EdgeTTSService] = None
rag_service: Optional[RAGService] = None
translation_service: Optional[TranslationService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and cleanup on shutdown"""
    global stt_service, llm_service, tts_service, rag_service, translation_service
    
    print("🚀 Initializing Voice Bot Services...")
    
    try:
        # Initialize Translation Service
        print("🌍 Loading Translation Service...")
        translation_service = TranslationService()
        print("✅ Translation Service Ready")
        
        # Initialize STT (Faster-Whisper)
        print("📝 Loading Faster-Whisper STT...")
        stt_service = FasterWhisperSTT(model_size="small", device="cpu", compute_type="int8")
        print("✅ STT Ready")
        
        # Initialize RAG (Qdrant + Sentence-Transformers)
        print("🔍 Loading RAG Service...")
        rag_service = RAGService(qdrant_path=":memory:")
        
        # Load sample IRCTC knowledge base
        sample_docs = [
            {"text": "IRCTC stands for Indian Railway Catering and Tourism Corporation. It handles online railway ticket booking."},
            {"text": "You can book train tickets online at www.irctc.co.in. Registration is required."},
            {"text": "Tatkal tickets can be booked one day in advance. Tatkal booking starts at 10 AM for AC classes and 11 AM for non-AC."},
            {"text": "To cancel a ticket, login to IRCTC, go to 'Booked Ticket History', and click on the ticket to cancel."},
            {"text": "PNR status can be checked on the IRCTC website or mobile app by entering your 10-digit PNR number."},
            {"text": "Senior citizens get a discount on train tickets. 40% for men above 60 and 50% for women above 58."},
            {"text": "You can file a TDR (Ticket Deposit Receipt) for refund if your train is delayed or cancelled."},
            {"text": "IRCTC customer care number is 14646 or 08044647999 for assistance."},
        ]
        rag_service.add_documents(sample_docs)
        print("✅ RAG Ready with sample IRCTC knowledge")
        
        # Load PDFs from data folder
        try:
            from load_pdfs import load_pdfs_into_rag
            load_pdfs_into_rag(rag_service, "data")
        except Exception as e:
            print(f"⚠️  Could not load PDFs: {e}")
            print("   Continuing with sample knowledge only...")

        
        # Initialize LLM (Mistral 7B)
        print("🤖 Loading Mistral LLM...")
        model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("⚠️  LLM will use fallback responses")
            llm_service = None
        else:
            try:
                llm_service = MistralLLM(model_path=model_path, n_gpu_layers=0)
                print("✅ LLM Ready")
            except Exception as llm_error:
                print(f"⚠️  LLM initialization failed: {llm_error}")
                print("⚠️  Continuing with fallback responses")
                llm_service = None
        
        # Initialize TTS (Edge-TTS)
        print("🔊 Loading Edge-TTS...")
        tts_service = EdgeTTSService(voice="en-US-ChristopherNeural")
        print("✅ TTS Ready")
        
        print("🎉 All services initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        print("⚠️  Some services may not be available")
    
    # Yield control to the application
    yield
    
    # Cleanup on shutdown
    print("🔄 Shutting down services...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="IRCTC Voice Assistant with WebSocket", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/voice-app.js")
async def serve_voice_app_js():
    """Serve the JavaScript file with no-cache headers"""
    with open("voice-app.js", "r", encoding="utf-8") as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/AIEnhancer_oneture_logo-removebg-preview.png")
async def serve_logo():
    """Serve the Oneture logo"""
    return FileResponse("AIEnhancer_oneture_logo-removebg-preview.png")

@app.get("/aws_logo.png")
async def serve_aws_logo():
    """Serve the AWS logo"""
    return FileResponse("aws_logo.png")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice conversation.
    
    Flow:
    1. Client sends audio chunks (PCM 16-bit, 16kHz)
    2. Server accumulates audio and transcribes with Faster-Whisper
    3. Server uses RAG to find relevant context
    4. Server generates response with Mistral LLM
    5. Server converts response to speech with Edge-TTS
    6. Server sends audio back to client
    """
    await websocket.accept()
    print("✅ WebSocket connected")
    
    # Send initial status
    await websocket.send_json({
        "type": "status",
        "message": "Connected! Start speaking..."
    })
    
    audio_buffer = bytearray()
    silence_counter = 0
    SILENCE_THRESHOLD = 6  # ~1.5 seconds at 16kHz with 4096 chunk size
    
    # Conversation State & History
    conversation_state = {"step": "idle", "data": {}}
    conversation_history = []
    
    # Flag to block audio processing while bot is responding
    is_processing = False
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive()
            
            if "bytes" in data:
                # IMPORTANT: Ignore all audio while bot is processing/speaking
                if is_processing:
                    continue  # Drop the audio chunk entirely
                
                # Audio chunk received
                audio_chunk = data["bytes"]
                audio_buffer.extend(audio_chunk)
                
                # Simple silence detection (check if audio is mostly zeros)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                avg_amplitude = np.abs(audio_array).mean()
                
                if avg_amplitude < 500:  # Silence threshold
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # If we have enough audio and detected silence, process it
                if len(audio_buffer) > 16000 * 2 and silence_counter >= SILENCE_THRESHOLD:
                    # Set processing flag to block further audio
                    is_processing = True
                    
                    print(f"🎤 Processing {len(audio_buffer)} bytes of audio...")
                    
                    # Send processing status
                    await websocket.send_json({
                        "type": "status",
                        "message": "Processing your speech..."
                    })
                    
                    # Process the audio
                    response_text, response_lang = await process_audio(audio_buffer, websocket, conversation_state, conversation_history)
                    
                    # Clear buffer immediately after processing to prevent re-processing
                    audio_buffer = bytearray()
                    silence_counter = 0
                    
                    # Update History
                    conversation_history.append({"role": "assistant", "content": response_text})
                    
                    # Send Bot Response
                    await websocket.send_json({
                        "type": "bot_response",
                        "text": response_text
                    })
                    
                    # Generate and send TTS audio
                    if tts_service:
                        try:
                            # Signal frontend to mute microphone
                            await websocket.send_json({
                                "type": "bot_speaking_start",
                                "message": "Bot is speaking..."
                            })
                            
                            audio_file = await tts_service.generate_speech(response_text, lang=response_lang)
                            
                            # Read audio file and send as bytes
                            with open(audio_file, "rb") as f:
                                audio_data = f.read()
                            
                            await websocket.send_bytes(audio_data)
                            
                            # Clean up temp file
                            os.remove(audio_file)
                            
                            # Calculate approximate playback duration (MP3 at ~128kbps)
                            # Add delay before allowing new audio
                            playback_duration = len(audio_data) / 16000  # Approximate
                            await asyncio.sleep(playback_duration + 0.5)  # Wait for playback + buffer
                            
                        except Exception as e:
                            print(f"❌ TTS Error: {e}")
                    
                    # Signal frontend that bot finished speaking
                    await websocket.send_json({
                        "type": "bot_speaking_end",
                        "message": "Bot finished speaking"
                    })
                    
                    # Keep history manageable
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]
                    
                    # Clear any audio that may have buffered during processing
                    audio_buffer = bytearray()
                    silence_counter = 0
                    
                    # Re-enable audio processing
                    is_processing = False
                    
                    # Send ready status
                    await websocket.send_json({
                        "type": "status",
                        "message": "Ready! Speak again..."
                    })
            
            elif "text" in data:
                # Text message received (control messages)
                message = json.loads(data["text"])
                
                if message.get("type") == "reset":
                    audio_buffer = bytearray()
                    silence_counter = 0
                    await websocket.send_json({
                        "type": "status",
                        "message": "Reset. Ready to listen..."
                    })
    
    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error: {str(e)}"
            })
        except:
            pass

async def process_audio(
    audio_bytes: bytearray, 
    websocket: Optional[WebSocket] = None,
    state: dict = None,
    history: list = None
) -> tuple[str, str]:
    """
    Process audio through the multilingual pipeline:
    STT -> Language Detection -> Intent Detection -> Response Generation
    Returns: (response_text, language_code)
    """
    try:
        # Step 1: Speech-to-Text
        if not stt_service:
            return "STT service not available. Please check model installation.", "en"
        
        print("🎤 Transcribing audio...")
        transcript = await stt_service.transcribe(bytes(audio_bytes))
        print(f"📝 Transcript: {transcript}")
        
        # Send transcript to client immediately
        if websocket and transcript and len(transcript.strip()) >= 1:
            try:
                await websocket.send_json({
                    "type": "user_transcript",
                    "text": transcript
                })
            except Exception as e:
                print(f"⚠️ Failed to send transcript update: {e}")

        if not transcript or len(transcript.strip()) < 1:
            return "I didn't catch that. Could you please repeat?", "en"
        
        # Step 2: Language Detection and Translation
        user_language = 'en'
        english_transcript = transcript
        
        if translation_service:
            # Translate to English for processing
            english_transcript, user_language = await translation_service.translate_to_english(transcript)
            lang_name = translation_service.get_language_name(user_language)
            print(f"🌍 Detected language: {lang_name} ({user_language})")
            if user_language != 'en':
                print(f"📝 English translation: {english_transcript}")
        
        if state is None: state = {"step": "idle", "data": {}}
        if history is None: history = []

        # Update history with user's latest query
        history.append({"role": "user", "content": english_transcript})
        
        user_text_lower = english_transcript.lower()
        original_lower = transcript.lower()

        # ============================================================
        # VAHAK LOGISTICS INTENT HANDLING
        # Enhanced for Hindi, Marathi, and English accuracy
        # ============================================================
        
        # ============================================================
        # DEMO QUESTIONS - SPECIFIC HANDLERS FOR SCREEN RECORDING
        # ============================================================
        
        # --- DEMO Q1: Hindi - "मुंबई से दिल्ली तक के truck book करो" ---
        # More flexible matching - check for Mumbai + Delhi anywhere + booking intent
        has_mumbai = any(p in original_lower or p in user_text_lower for p in [
            "मुंबई", "मुम्बई", "mumbai", "मुम्बए"
        ])
        has_delhi = any(p in original_lower or p in user_text_lower for p in [
            "दिल्ली", "delhi", "दिल्लि", "दिल्लि"
        ])
        has_hindi_booking = any(p in original_lower or p in user_text_lower for p in [
            "बुक", "बुकिंग", "करो", "करना", "चाहिए", "चाहिये", "book", "booking"
        ])
        has_truck_hindi = any(p in original_lower or p in user_text_lower for p in [
            "ट्रक", "truck", "गाड़ी", "गाडी", "ट्रॅक"
        ])
        
        # Check if it's a Hindi Mumbai-Delhi truck booking
        is_mumbai_delhi_booking = has_mumbai and has_delhi and (has_hindi_booking or has_truck_hindi)
        
        # Also check for Devanagari script to confirm Hindi
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in transcript)
        
        if is_mumbai_delhi_booking and has_devanagari:
            booking_id = f"VHK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            state["booking_id"] = booking_id
            
            # Response in Hindi
            response_text = f"बहुत बढ़िया! मैंने मुंबई से दिल्ली तक एक ट्रक की बुकिंग शुरू कर दी है। आपकी बुकिंग ID है: {booking_id}। अनुमानित लागत ₹45,000 से ₹55,000 है। वाहक पार्टनर जल्द ही पिकअप टाइम कन्फर्म करने के लिए आपसे संपर्क करेगा। क्या आपको कुछ और चाहिए?"
            state["last_bot_response"] = response_text
            return response_text, 'hi'
        
        # --- DEMO Q2: English - "I wanted to book a truck" ---
        english_truck_booking_patterns = [
            "i wanted to book a truck",
            "want to book a truck", 
            "i want to book a truck",
            "wanted to book a truck",
            "book a truck"
        ]
        
        is_english_truck_booking = any(p in user_text_lower for p in english_truck_booking_patterns)
        
        if is_english_truck_booking:
            booking_id = f"VHK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            state["booking_id"] = booking_id
            
            response_text = f"Great! I'd be happy to help you book a truck. Your provisional Booking ID is {booking_id}. To complete the booking, please tell me: 1) Pickup city, 2) Delivery city, and 3) Truck type if you have a preference (20ft, 22ft, or 32ft). What are your pickup and delivery locations?"
            state["last_bot_response"] = response_text
            return response_text, 'en'
        
        # --- DEMO Q3: Marathi - "मराठी मध्ये बोला" ---
        marathi_request_patterns = [
            "मराठी मध्ये बोला", "marathi madhe bola", "मराठीत बोला",
            "मराठी मध्ये", "marathi madhe", "मराठीत", "in marathi",
            "speak marathi", "speak in marathi"
        ]
        
        is_marathi_request = any(p in original_lower for p in marathi_request_patterns) or \
                            any(p in user_text_lower for p in marathi_request_patterns)
        
        if is_marathi_request:
            # Response in Marathi
            response_text = "नमस्कार! मी आता मराठीत बोलतो. मी तुम्हाला वाहक लॉजिस्टिक्समध्ये मदत करू शकतो. तुम्हाला ट्रक बुकिंग, रेट इन्क्वायरी, शिपमेंट ट्रॅकिंग किंवा इतर कोणत्याही माहितीसाठी मदत हवी असेल तर मला सांगा. तुम्हाला काय मदत हवी आहे?"
            state["last_bot_response"] = response_text
            state["preferred_language"] = "mr"
            return response_text, 'mr'
        
        # ============================================================
        # GENERAL INTENT HANDLING (if demo questions don't match)
        # ============================================================
        
        # --- 1. RATE INQUIRY ---
        # English keywords
        rate_keywords_en = ["rate", "price", "cost", "quote", "charge", "how much", "pricing", "fare", "tariff"]
        # Hindi keywords (Romanized + Devanagari patterns)
        rate_keywords_hi = ["kitna", "kharcha", "bhav", "daam", "paisa", "rupees", "rs", "kimat", "dar"]
        # Marathi keywords (Romanized)
        rate_keywords_mr = ["kiti", "kharch", "dar", "bhav", "rupaye"]
        
        # Devanagari patterns for rate inquiry
        rate_devanagari = ["रेट", "भाव", "खर्च", "कीमत", "दाम", "पैसा", "कितना", "किती", "दर", "किती खर्च"]
        
        # City keywords for route detection
        route_keywords = ["mumbai", "delhi", "pune", "bangalore", "bengaluru", "chennai", "hyderabad", 
                         "kolkata", "jaipur", "ahmedabad", "lucknow", "chandigarh", "indore", "nagpur", 
                         "surat", "vadodara", "nashik", "aurangabad", "thane", "navi mumbai"]
        # Hindi city names
        route_hindi = ["मुंबई", "दिल्ली", "पुणे", "बेंगलुरु", "चेन्नई", "हैदराबाद", "कोलकाता", "जयपुर"]
        
        is_rate_inquiry = any(kw in user_text_lower for kw in rate_keywords_en + rate_keywords_hi + rate_keywords_mr) or \
                          any(kw in original_lower for kw in rate_devanagari) or \
                          any(kw in transcript for kw in rate_devanagari)
        
        if is_rate_inquiry:
            # Extract route info if present
            from_city = None
            to_city = None
            # Check English city names
            for city in route_keywords:
                if city in user_text_lower:
                    if from_city is None:
                        from_city = city.capitalize()
                    else:
                        to_city = city.capitalize()
                        break
            # Check Hindi city names in original
            if not from_city:
                for city in route_hindi:
                    if city in transcript:
                        city_map = {"मुंबई": "Mumbai", "दिल्ली": "Delhi", "पुणे": "Pune", 
                                   "बेंगलुरु": "Bangalore", "चेन्नई": "Chennai", 
                                   "हैदराबाद": "Hyderabad", "कोलकाता": "Kolkata", "जयपुर": "Jaipur"}
                        if from_city is None:
                            from_city = city_map.get(city, city)
                        else:
                            to_city = city_map.get(city, city)
                            break
            
            if from_city and to_city:
                response_text = f"The approximate rate for a truck from {from_city} to {to_city} is ₹35,000 to ₹55,000 depending on the truck type. For a 20 feet container it's around ₹40,000, and for a 32 feet truck it's approximately ₹55,000. Toll charges are extra. Would you like to book a truck for this route?"
            elif from_city:
                response_text = f"For transportation from {from_city}, our rates start from ₹8,000 for nearby cities and go up to ₹65,000 for long-distance routes. Please tell me the destination city so I can give you an exact quote."
            else:
                response_text = "Our truck rates depend on the route and truck type. For example, Mumbai to Delhi costs ₹45,000-65,000, Mumbai to Pune costs ₹8,000-15,000. Please tell me your pickup and delivery locations for an exact quote."
            
            state["last_bot_response"] = response_text
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # --- 2. TRUCK BOOKING ---
        # English booking keywords
        booking_keywords_en = ["book", "booking", "need", "want", "require", "arrange", "hire", "get me"]
        # Hindi booking keywords (Romanized)
        booking_keywords_hi = ["chahiye", "chahte", "karna", "karwana", "karwao", "kardo", "mangwana", "lagwana"]
        # Marathi booking keywords (Romanized)
        booking_keywords_mr = ["karaycha", "karaychi", "pahije", "hava", "havi", "have", "lage", "lagte"]
        
        # Devanagari booking patterns
        booking_devanagari = ["बुक", "चाहिए", "करना है", "करायचा", "करायची", "हवा आहे", "हवी आहे", 
                             "हवा", "लागेल", "पाहिजे", "बुकिंग"]
        
        # Truck keywords in all languages
        truck_keywords_en = ["truck", "container", "tempo", "lorry", "vehicle", "trailer", "transport"]
        truck_devanagari = ["ट्रक", "कंटेनर", "गाड़ी", "टेम्पो", "ट्रेलर", "वाहन", "ट्रॅक"]
        
        is_booking = any(kw in user_text_lower for kw in booking_keywords_en + booking_keywords_hi + booking_keywords_mr) or \
                     any(kw in original_lower for kw in booking_devanagari) or \
                     any(kw in transcript for kw in booking_devanagari)
        has_truck = any(kw in user_text_lower for kw in truck_keywords_en) or \
                    any(kw in original_lower for kw in truck_devanagari) or \
                    any(kw in transcript for kw in truck_devanagari)
        
        if is_booking and has_truck:
            # Extract details
            from_city = None
            to_city = None
            for city in route_keywords:
                if city in user_text_lower:
                    if from_city is None:
                        from_city = city.capitalize()
                    else:
                        to_city = city.capitalize()
                        break
            
            # Check for truck size (works with Hindi numbers too)
            truck_size = "standard truck"
            if "20" in english_transcript or "२०" in transcript or "बीस" in transcript:
                truck_size = "20 feet container"
            elif "22" in english_transcript or "२२" in transcript or "बाईस" in transcript:
                truck_size = "22 feet truck"
            elif "24" in english_transcript or "२४" in transcript or "चौबीस" in transcript:
                truck_size = "24 feet container"
            elif "32" in english_transcript or "३२" in transcript or "बत्तीस" in transcript:
                truck_size = "32 feet multi-axle truck"
            
            # Check for load weight (Hindi: टन, Marathi: टन)
            load_match = re.search(r"(\d+)\s*(ton|tons|टन)", user_text_lower + original_lower)
            load_info = f" for {load_match.group(1)} tons load" if load_match else ""
            
            booking_id = f"VHK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            
            if from_city and to_city:
                response_text = f"Great! I have initiated a booking for a {truck_size}{load_info} from {from_city} to {to_city}. Your Booking ID is {booking_id}. Estimated cost is ₹35,000 to ₹45,000. A Vahak partner will contact you shortly to confirm the pickup time. Is there anything else you need?"
            elif from_city:
                response_text = f"I can help you book a {truck_size}{load_info} from {from_city}. Please provide the destination city to complete the booking."
            else:
                response_text = f"I can help you book a {truck_size}{load_info}. Your provisional Booking ID is {booking_id}. Please provide the pickup and delivery cities to proceed."
            
            state["last_bot_response"] = response_text
            state["booking_id"] = booking_id
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # --- 3. SHIPMENT TRACKING ---
        # English tracking keywords
        tracking_keywords_en = ["track", "tracking", "where", "status", "location", "find", "check", "locate"]
        # Hindi tracking keywords (Romanized)
        tracking_keywords_hi = ["kahan", "kidhar", "pata", "status", "sthiti"]
        # Marathi tracking keywords (Romanized)
        tracking_keywords_mr = ["kuthe", "kuthay", "status", "track"]
        
        # Devanagari tracking patterns
        tracking_devanagari = ["ट्रैक", "ट्रॅक", "कहां", "कहाँ", "कुठे", "कहाँ है", "कुठे आहे", 
                              "स्थिति", "पता", "ट्रैकिंग", "ट्रॅकिंग"]
        
        # Shipment keywords
        shipment_keywords_en = ["shipment", "order", "truck", "delivery", "parcel", "package", "consignment", "goods", "cargo"]
        shipment_devanagari = ["माल", "शिपमेंट", "ऑर्डर", "डिलिवरी", "सामान", "पार्सल"]
        
        is_tracking = any(kw in user_text_lower for kw in tracking_keywords_en + tracking_keywords_hi + tracking_keywords_mr) or \
                      any(kw in original_lower for kw in tracking_devanagari) or \
                      any(kw in transcript for kw in tracking_devanagari)
        has_shipment = any(kw in user_text_lower for kw in shipment_keywords_en) or \
                       any(kw in original_lower for kw in shipment_devanagari) or \
                       any(kw in transcript for kw in shipment_devanagari)
        
        # Extract order number - support various formats
        order_match = re.search(r"(?:order|booking|id|number|नंबर|no\.?|#)\s*:?\s*(\w*[-]?\d+)", user_text_lower + original_lower, re.IGNORECASE)
        if not order_match:
            order_match = re.search(r"\b(\d{4,})\b", english_transcript + transcript)
        
        if is_tracking or (has_shipment and ("where" in user_text_lower or "status" in user_text_lower or "कहां" in original_lower or "कुठे" in original_lower)):
            if order_match:
                order_id = order_match.group(1).upper()
                response_text = f"Tracking your shipment with Order ID {order_id}:\n\n📍 Current Status: In Transit\n🚛 Truck Number: MH-04-AB-1234\n📅 Pickup: Mumbai - Completed (Yesterday 2:30 PM)\n🎯 Destination: Delhi\n⏰ Expected Delivery: Tomorrow by 6:00 PM\n📱 Driver Contact: +91-98765-43210\n\nYour shipment is currently near Jaipur and on schedule."
            else:
                response_text = "I can help you track your shipment. Please provide your Order ID or Booking Number and I'll give you the real-time status."
            
            state["last_bot_response"] = response_text
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # --- 4. GENERAL INFORMATION - TRUCK TYPES ---
        # English truck type queries
        truck_type_en = ["truck type", "which truck", "types of truck", "available truck", "what truck", "truck options"]
        # Hindi queries (Romanized)
        truck_type_hi = ["kaun sa truck", "kaun se truck", "kaunsa", "kaunse", "konsa truck"]
        # Marathi queries (Romanized)
        truck_type_mr = ["konata truck", "konate truck", "konata", "konate"]
        # Devanagari patterns
        truck_type_devanagari = ["कौन से ट्रक", "कौन सा ट्रक", "कौन-कौन", "प्रकारचे ट्रक", "कोणत्या", 
                                "कोणत्या प्रकारचे", "कोणते ट्रक", "ट्रक प्रकार"]
        
        is_truck_type_query = any(kw in user_text_lower for kw in truck_type_en + truck_type_hi + truck_type_mr) or \
                              any(kw in original_lower for kw in truck_type_devanagari) or \
                              any(kw in transcript for kw in truck_type_devanagari)
        
        if is_truck_type_query:
            response_text = """We have the following truck types available:

📦 Container Trucks (Closed Body):
• 14 feet (4-5 ton) - Small cargo
• 20 feet (8-9 ton) - Standard loads
• 24 feet (12-14 ton) - Large shipments
• 32 feet (20-25 ton) - Heavy cargo

🚛 Open Body Trucks:
• 17 feet (6-7 ton)
• 22 feet (10-12 ton)

🌡️ Specialized: Reefer trucks, Tankers, Flatbed trailers

Which type would you like to book?"""
            state["last_bot_response"] = response_text
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # --- 5. GENERAL INFORMATION - ABOUT VAHAK ---
        # English keywords
        about_keywords_en = ["vahak", "about", "service", "what do you", "tell me about", "company", "who are you"]
        # Hindi keywords (Romanized)
        about_keywords_hi = ["seva", "baare", "batao", "bataye", "company", "kya karte"]
        # Marathi keywords (Romanized)
        about_keywords_mr = ["seva", "sangta", "sanga", "kahi sanga", "company"]
        # Devanagari patterns
        about_devanagari = ["वाहक", "सेवा", "सांगा", "बताएं", "बताओ", "बारे में", "कंपनी", "क्या करते"]
        
        is_about_query = any(kw in user_text_lower for kw in about_keywords_en + about_keywords_hi + about_keywords_mr) or \
                        any(kw in original_lower for kw in about_devanagari) or \
                        any(kw in transcript for kw in about_devanagari)
        
        if is_about_query:
            response_text = """Welcome to Vahak Logistics! 🚛

We are India's leading digital platform for truck booking. Our services include:

✅ Full Truck Load (FTL) - Book entire trucks
✅ Part Load (PTL) - Share truck space, save costs
✅ Express Delivery - Priority shipping
✅ Real-time GPS Tracking
✅ Verified transporters across 500+ cities
✅ 24/7 customer support

We cover all major routes across India including Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, and more.

How can I help you today?"""
            state["last_bot_response"] = response_text
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # --- 6. ROUTES AVAILABLE ---
        # English keywords
        route_keywords_en = ["route", "coverage", "where", "which cities", "available cities", "areas"]
        # Hindi keywords (Romanized)
        route_keywords_hi = ["rasta", "jagah", "shahar", "kahan kahan"]
        # Marathi keywords (Romanized)
        route_keywords_mr = ["marg", "kuthe kuthe", "shahare"]
        # Devanagari patterns
        route_devanagari = ["रूट", "कौन से", "कोणते", "उपलब्ध", "कहाँ कहाँ", "शहर", "कुठे कुठे"]
        
        is_route_query = any(kw in user_text_lower for kw in route_keywords_en + route_keywords_hi + route_keywords_mr) or \
                        any(kw in original_lower for kw in route_devanagari) or \
                        any(kw in transcript for kw in route_devanagari)
        
        if is_route_query:
            response_text = """We cover all major routes across India:

🏙️ Metro City Routes (1-2 days):
• Mumbai ↔ Delhi
• Bangalore ↔ Chennai
• Mumbai ↔ Pune
• Delhi ↔ Jaipur

🌆 Tier-2 City Routes (2-3 days):
• Ahmedabad, Surat, Lucknow
• Chandigarh, Indore, Nagpur

📍 Pan-India Coverage:
• 500+ cities served
• 10,000+ verified transporters
• Northeast and remote areas covered

Which route are you interested in?"""
            state["last_bot_response"] = response_text
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # --- 7. GREETING / HELP ---
        # English greetings
        greeting_en = ["hello", "hi", "hey", "help", "assist", "good morning", "good evening"]
        # Hindi greetings (Romanized)
        greeting_hi = ["namaste", "namaskar", "namasте", "madad", "sahayata"]
        # Marathi greetings (Romanized)
        greeting_mr = ["namaskar", "madad", "sahayya"]
        # Devanagari patterns
        greeting_devanagari = ["नमस्ते", "नमस्कार", "मदद", "सहायता", "हेलो", "हाय"]
        
        is_greeting = any(word in user_text_lower.split() for word in greeting_en + greeting_hi + greeting_mr) or \
                     any(kw in original_lower for kw in greeting_devanagari) or \
                     any(kw in transcript for kw in greeting_devanagari)
        
        if is_greeting:
            response_text = """Namaste! Welcome to Vahak Logistics. 🙏

I can help you with:
1️⃣ Rate Inquiry - Get price quotes for any route
2️⃣ Truck Booking - Book trucks instantly
3️⃣ Shipment Tracking - Track your orders
4️⃣ General Information - Learn about our services

What would you like to do today?"""
            state["last_bot_response"] = response_text
            if translation_service and user_language != 'en':
                response_text = await translation_service.translate_from_english(response_text, user_language)
            return response_text, user_language

        # ============================================================
        # FALLBACK - Use RAG + LLM for other queries
        # ============================================================
        
        context_docs = []
        if rag_service:
            print("🔍 Searching knowledge base...")
            context_docs = rag_service.search(english_transcript, limit=3)
            print(f"📚 Found {len(context_docs)} relevant documents")
        
        english_response = ""
        if llm_service:
            print("🤖 Generating LLM response...")
            
            system_prompt = "You are Vahak's helpful logistics assistant. Help customers with truck booking, rate inquiries, shipment tracking, and general logistics questions. Be concise, friendly, and helpful. Always respond in a professional manner."
            
            context_str = "\n".join([doc['text'] for doc in context_docs]) if context_docs else ""
            
            prompt = f"<s>[INST] {system_prompt}\n\n"
            if context_str:
                prompt += f"Context:\n{context_str}\n\n"
            
            recent_history = history[-6:]
            for msg in recent_history:
                role = "User" if msg['role'] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            
            prompt += "[/INST]"
            
            english_response = llm_service.generate_response(prompt, max_tokens=150)
            print(f"💬 English response: {english_response}")
            state["last_bot_response"] = english_response
        else:
            # Fallback response
            if context_docs and len(context_docs) > 0:
                english_response = f"Based on your query: {context_docs[0]['text']}"
            else:
                english_response = "I'm here to help with Vahak logistics services. You can ask me about truck rates, booking, tracking shipments, or our services. How can I assist you?"
            state["last_bot_response"] = english_response
        
        # Translate response back to user's language
        final_response = english_response
        if translation_service and user_language != 'en':
            final_response = await translation_service.translate_from_english(english_response, user_language)
            print(f"🌍 Translated response to {user_language}: {final_response}")
        
        return final_response, user_language
    
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return "I encountered an error processing your request. Please try again.", "en"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    translation_status = "not loaded"
    if translation_service:
        translation_status = "ready (with Google Translate)" if translation_service.translation_available else "ready (basic detection only)"
    
    return {
        "status": "healthy",
        "services": {
            "stt": "ready" if stt_service else "not loaded",
            "llm": "ready" if llm_service else "not loaded (using fallback)",
            "tts": "ready" if tts_service else "not loaded",
            "rag": "ready" if rag_service else "not loaded",
            "translation": translation_status
        },
        "tech_stack": {
            "stt": "Faster-Whisper (Multilingual)",
            "llm": "Mistral-7B-Instruct (GGUF)",
            "tts": "Edge-TTS",
            "rag": "Qdrant + Sentence-Transformers",
            "translation": "Google Translate API (optional)",
            "backend": "FastAPI + WebSocket"
        },
        "supported_languages": ["Hindi", "English", "Marathi", "Tamil", "Telugu", "Bengali", "and more..."]
    }

@app.get("/test-stt")
async def test_stt():
    """Test endpoint to verify STT service"""
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT service not initialized")
    
    return {
        "status": "STT service is ready",
        "model": "Faster-Whisper (small)"
    }

@app.get("/test-rag")
async def test_rag():
    """Test endpoint to verify RAG service"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # Test search
    results = rag_service.search("How to book tickets?", limit=2)
    
    return {
        "status": "RAG service is ready",
        "test_query": "How to book tickets?",
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting IRCTC Voice Assistant Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🎤 WebSocket endpoint: ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)

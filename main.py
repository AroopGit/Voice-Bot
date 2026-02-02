from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import json
from langdetect import detect, LangDetectException
import re

app = FastAPI(title="Multilingual Voice Conversation POC")

# CORS middleware for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Inference API (100% FREE - No API key needed for public models)
# Using Mistral-7B-Instruct which is completely free
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# Language code mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'hi': 'Hindi',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'ru': 'Russian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ur': 'Urdu',
}

class TranscriptRequest(BaseModel):
    transcript: str

class ProcessResponse(BaseModel):
    language_id: str
    bot_reply: str
    subject: str
    concerns: list[str]
    queries: list[str]

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/app.js")
async def serve_app_js():
    """Serve the JavaScript file with no-cache headers"""
    from fastapi.responses import Response
    
    with open("app.js", "r", encoding="utf-8") as f:
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

@app.get("/voice-app.js")
async def serve_voice_app_js():
    """Serve the new JavaScript file with no-cache headers"""
    from fastapi.responses import Response
    
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

def detect_language(text: str) -> str:
    """
    Detect language using langdetect library (100% free, offline)
    Enhanced for better Hindi and English detection
    """
    try:
        # Check for Hindi characters (Devanagari script)
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        if hindi_chars > len(text) * 0.3:  # If 30%+ are Hindi characters
            return 'Hindi'
        
        # Use langdetect for other languages
        lang_code = detect(text)
        detected = LANGUAGE_NAMES.get(lang_code, 'English')
        
        # Verify English detection
        if lang_code == 'en':
            return 'English'
        elif lang_code == 'hi':
            return 'Hindi'
        
        return detected
    except LangDetectException:
        # Fallback: check for Hindi characters
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        if hindi_chars > 0:
            return 'Hindi'
        return 'English'

def extract_json_from_text(text: str) -> dict:
    """Extract JSON object from text response"""
    # Try to find JSON object in the response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None

async def query_huggingface(prompt: str, max_retries: int = 3) -> str:
    """
    Query Hugging Face Inference API (100% FREE)
    No API key needed for public models
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    HF_API_URL,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 500,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "do_sample": True,
                        }
                    }
                )
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail="Model is currently loading. Please try again in a few seconds."
                        )
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    return result.get('generated_text', '')
                
                return str(result)
                
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise HTTPException(
                    status_code=500,
                    detail=f"Error calling Hugging Face API: {str(e)}"
                )

def parse_ai_response(response_text: str, detected_language: str) -> dict:
    """Parse AI response and extract structured data"""
    # Try to extract JSON first
    json_data = extract_json_from_text(response_text)
    if json_data:
        return json_data
    
    # Fallback: Manual parsing
    lines = response_text.split('\n')
    result = {
        'language_id': detected_language,
        'bot_reply': '',
        'subject': 'General conversation',
        'concerns': [],
        'queries': []
    }
    
    # Extract bot reply (usually the first substantial text)
    for line in lines:
        line = line.strip()
        if line and not line.startswith('{') and not line.startswith('['):
            if not result['bot_reply']:
                result['bot_reply'] = line
            elif len(result['bot_reply']) < 200:
                result['bot_reply'] += ' ' + line
    
    # Extract concerns (look for keywords)
    concern_keywords = ['problem', 'issue', 'trouble', 'concern', 'worry', 'error', 
                        'problema', 'समस्या', 'مشكلة', 'problème']
    for line in lines:
        for keyword in concern_keywords:
            if keyword.lower() in line.lower():
                result['concerns'].append(line.strip())
                break
    
    # Extract queries (look for question marks)
    for line in lines:
        if '?' in line:
            result['queries'].append(line.strip())
    
    return result

@app.post("/process", response_model=ProcessResponse)
async def process_transcript(request: TranscriptRequest):
    """
    Process user transcript using Hugging Face's FREE models
    - Language Detection: langdetect (offline, free)
    - Response Generation: Mistral-7B-Instruct (free Hugging Face API)
    """
    try:
        # Step 1: Detect language (offline, instant)
        detected_language = detect_language(request.transcript)
        
        # Step 2: Create prompt for Hugging Face model
        prompt = f"""<s>[INST] You are a helpful multilingual assistant. A user said: "{request.transcript}"

The user is speaking in {detected_language}. Please:
1. Respond naturally in {detected_language}
2. Identify the main subject/topic
3. List any concerns or problems mentioned
4. List any questions asked

Respond in this format:
REPLY: [your response in {detected_language}]
SUBJECT: [main topic]
CONCERNS: [list concerns separated by semicolons, or "none"]
QUERIES: [list questions separated by semicolons, or "none"]
[/INST]"""

        # Step 3: Query Hugging Face (FREE)
        ai_response = await query_huggingface(prompt)
        
        # Step 4: Parse response
        result = {
            'language_id': detected_language,
            'bot_reply': '',
            'subject': 'General conversation',
            'concerns': [],
            'queries': []
        }
        
        # Extract structured data from response
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('REPLY:'):
                result['bot_reply'] = line.replace('REPLY:', '').strip()
            elif line.startswith('SUBJECT:'):
                result['subject'] = line.replace('SUBJECT:', '').strip()
            elif line.startswith('CONCERNS:'):
                concerns_text = line.replace('CONCERNS:', '').strip()
                if concerns_text.lower() != 'none':
                    result['concerns'] = [c.strip() for c in concerns_text.split(';') if c.strip()]
            elif line.startswith('QUERIES:'):
                queries_text = line.replace('QUERIES:', '').strip()
                if queries_text.lower() != 'none':
                    result['queries'] = [q.strip() for q in queries_text.split(';') if q.strip()]
        
        # Fallback if parsing failed - use NLP on original transcript
        if not result['bot_reply']:
            # Use the full response as reply
            result['bot_reply'] = ai_response.split('[/INST]')[-1].strip()
            if not result['bot_reply']:
                # Generate a simple response in the detected language
                if detected_language == 'Hindi':
                    result['bot_reply'] = f"मैं समझ गया। आपने कहा: {request.transcript}"
                else:
                    result['bot_reply'] = f"I understand. You said: {request.transcript}"
        
        # Enhanced subject extraction from transcript if not found
        if result['subject'] == 'General conversation' or not result['subject']:
            # Try to extract meaningful subject from transcript
            words = request.transcript.split()
            if len(words) > 3:
                # Take first meaningful phrase (skip common words)
                skip_words = {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'am', 'are', 'मैं', 'मुझे', 'मेरा'}
                meaningful_words = [w for w in words if w.lower() not in skip_words]
                if meaningful_words:
                    result['subject'] = ' '.join(meaningful_words[:4])
                else:
                    result['subject'] = ' '.join(words[:5])
        
        # Enhanced query detection from transcript if not found by AI
        if not result['queries']:
            # Check for question marks or question words
            question_words_en = ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']
            question_words_hi = ['क्या', 'कब', 'कहाँ', 'कौन', 'क्यों', 'कैसे']
            
            if '?' in request.transcript:
                result['queries'].append(request.transcript)
            else:
                # Check if starts with question word
                first_word = request.transcript.split()[0].lower() if request.transcript.split() else ''
                if first_word in question_words_en or first_word in question_words_hi:
                    result['queries'].append(request.transcript)
        
        # Enhanced concern detection from transcript
        if not result['concerns']:
            # Look for problem/issue keywords
            concern_keywords_en = ['problem', 'issue', 'trouble', 'error', 'wrong', 'not working', 'broken', 'help', 'stuck']
            concern_keywords_hi = ['समस्या', 'परेशानी', 'गलत', 'मदद', 'काम नहीं']
            
            transcript_lower = request.transcript.lower()
            for keyword in concern_keywords_en + concern_keywords_hi:
                if keyword in transcript_lower:
                    result['concerns'].append(f"User mentioned: {keyword}")
                    break
        
        return ProcessResponse(**result)
        
    except Exception as e:
        # Fallback response if AI fails
        detected_language = detect_language(request.transcript)
        return ProcessResponse(
            language_id=detected_language,
            bot_reply=f"I heard you say: {request.transcript}. How can I help you?",
            subject="General inquiry",
            concerns=[],
            queries=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_backend": "Hugging Face (Free)",
        "model": "Mistral-7B-Instruct-v0.2",
        "language_detection": "langdetect (offline)"
    }

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000)

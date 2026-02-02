@echo off
REM Start the Multilingual Voice Bot Server
REM No API key needed - 100% FREE!

echo ============================================
echo Multilingual Voice Conversation POC
echo ============================================
echo.
echo [✓] Using FREE Hugging Face API
echo [✓] Model: Mistral-7B-Instruct-v0.2
echo [✓] No API key required!
echo.
echo Starting FastAPI server...
echo.
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python main.py

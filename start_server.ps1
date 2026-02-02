# Start the Multilingual Voice Bot Server
# No API key needed - 100% FREE!

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Multilingual Voice Conversation POC" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[✓] Using FREE Hugging Face API" -ForegroundColor Green
Write-Host "[✓] Model: Mistral-7B-Instruct-v0.2" -ForegroundColor Green
Write-Host "[✓] No API key required!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python main.py

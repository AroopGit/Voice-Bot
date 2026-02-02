# Quick Start Script for Voice Bot with PDF RAG

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VOICE BOT WITH PDF RAG - QUICK START" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if data folder exists
if (Test-Path "data") {
    $pdfCount = (Get-ChildItem "data\*.pdf").Count
    Write-Host "[OK] data/ folder exists" -ForegroundColor Green
    Write-Host "     Found $pdfCount PDF file(s)" -ForegroundColor Cyan
} else {
    Write-Host "[WARN] data/ folder not found" -ForegroundColor Yellow
    Write-Host "       Creating data/ folder..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
    Write-Host "[OK] Created data/ folder" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Voice Bot Server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Watch for these messages:" -ForegroundColor Cyan
Write-Host "  - Loading PDF DOCUMENTS INTO RAG" -ForegroundColor Gray
Write-Host "  - Extracted X chunks from [your PDF]" -ForegroundColor Gray
Write-Host "  - All PDF documents loaded successfully!" -ForegroundColor Gray
Write-Host ""
Write-Host "Then open: http://localhost:8000" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start the server
python main_websocket.py

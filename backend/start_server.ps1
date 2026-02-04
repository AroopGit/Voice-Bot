# SwiftShip Voice Bot - PowerShell Startup Script
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "SwiftShip Voice Bot - Startup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
$activateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}

# Check if fastapi is installed
$fastapiCheck = pip show fastapi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "Creating from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Please edit .env and add your GROQ_API_KEY" -ForegroundColor Red
    Write-Host ""
}

# Check if Qdrant is running
$qdrantCheck = Test-NetConnection -ComputerName localhost -Port 6333 -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
if (-not $qdrantCheck.TcpTestSucceeded) {
    Write-Host ""
    Write-Host "WARNING: Qdrant is not running on localhost:6333" -ForegroundColor Yellow
    Write-Host "Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Start the server
python main.py

# Voice Bot Verification Script
# Run this to check all components

Write-Host "🔍 VOICE BOT SYSTEM VERIFICATION" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Check 1: Python Installation
Write-Host "✓ Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Python not found!" -ForegroundColor Red
    $errors++
}

# Check 2: Required Python Packages
Write-Host ""
Write-Host "✓ Checking Python packages..." -ForegroundColor Yellow

$requiredPackages = @(
    "fastapi",
    "uvicorn",
    "faster-whisper",
    "qdrant-client",
    "sentence-transformers",
    "llama-cpp-python",
    "edge-tts",
    "websockets",
    "numpy"
)

foreach ($package in $requiredPackages) {
    $installed = pip show $package 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ $package installed" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $package NOT installed" -ForegroundColor Red
        $errors++
    }
}

# Check 3: Model Files
Write-Host ""
Write-Host "✓ Checking model files..." -ForegroundColor Yellow

$modelPath = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if (Test-Path $modelPath) {
    $size = (Get-Item $modelPath).Length / 1GB
    Write-Host "  ✅ Mistral model found ($([math]::Round($size, 2)) GB)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Mistral model NOT found at $modelPath" -ForegroundColor Yellow
    Write-Host "     System will use fallback responses" -ForegroundColor Yellow
    $warnings++
}

# Check 4: Project Files
Write-Host ""
Write-Host "✓ Checking project files..." -ForegroundColor Yellow

$requiredFiles = @(
    "main_websocket.py",
    "index.html",
    "voice-app.js",
    "requirements.txt",
    "app\services\stt.py",
    "app\services\llm.py",
    "app\services\tts.py",
    "app\services\rag.py"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file NOT found" -ForegroundColor Red
        $errors++
    }
}

# Check 5: Port Availability
Write-Host ""
Write-Host "✓ Checking port 8000..." -ForegroundColor Yellow

$portInUse = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "  ⚠️  Port 8000 is already in use" -ForegroundColor Yellow
    Write-Host "     You may need to stop the existing server" -ForegroundColor Yellow
    $warnings++
} else {
    Write-Host "  ✅ Port 8000 is available" -ForegroundColor Green
}

# Check 6: Network Connectivity
Write-Host ""
Write-Host "✓ Checking network connectivity..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "https://www.google.com" -TimeoutSec 5 -UseBasicParsing
    Write-Host "  ✅ Internet connection available" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  No internet connection (required for Edge-TTS)" -ForegroundColor Yellow
    $warnings++
}

# Summary
Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✅ ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Ready to start the server!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run: python main_websocket.py" -ForegroundColor Cyan
} elseif ($errors -eq 0) {
    Write-Host "⚠️  $warnings WARNING(S) - System can run with limitations" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🚀 You can start the server, but some features may not work" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run: python main_websocket.py" -ForegroundColor Cyan
} else {
    Write-Host "❌ $errors ERROR(S) and $warnings WARNING(S) found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the errors before starting the server" -ForegroundColor Red
    Write-Host ""
    Write-Host "To install missing packages:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Offer to start server if no errors
if ($errors -eq 0) {
    $response = Read-Host "Would you like to start the server now? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host ""
        Write-Host "[START] Starting Voice Bot Server..." -ForegroundColor Green
        Write-Host ""
        python main_websocket.py
    }
}

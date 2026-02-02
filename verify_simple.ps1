# Voice Bot Verification Script (Simple Version)
# Run this to check all components

Write-Host "[CHECK] VOICE BOT SYSTEM VERIFICATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Check 1: Python Installation
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python not found!" -ForegroundColor Red
    $errors++
}

# Check 2: Required Python Packages
Write-Host ""
Write-Host "[2/6] Checking Python packages..." -ForegroundColor Yellow

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

$packagesOK = 0
foreach ($package in $requiredPackages) {
    $installed = pip show $package 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $packagesOK++
    }
}

Write-Host "  [INFO] $packagesOK / $($requiredPackages.Count) packages installed" -ForegroundColor Cyan
if ($packagesOK -lt $requiredPackages.Count) {
    Write-Host "  [WARN] Some packages missing. Run: pip install -r requirements_complete.txt" -ForegroundColor Yellow
    $warnings++
}

# Check 3: Model Files
Write-Host ""
Write-Host "[3/6] Checking model files..." -ForegroundColor Yellow

$modelPath = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if (Test-Path $modelPath) {
    $size = (Get-Item $modelPath).Length / 1GB
    Write-Host "  [OK] Mistral model found ($([math]::Round($size, 2)) GB)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Mistral model NOT found at $modelPath" -ForegroundColor Yellow
    Write-Host "        System will use fallback responses" -ForegroundColor Yellow
    $warnings++
}

# Check 4: Project Files
Write-Host ""
Write-Host "[4/6] Checking project files..." -ForegroundColor Yellow

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

$filesOK = 0
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        $filesOK++
    }
}

Write-Host "  [INFO] $filesOK / $($requiredFiles.Count) files found" -ForegroundColor Cyan
if ($filesOK -lt $requiredFiles.Count) {
    Write-Host "  [ERROR] Some project files missing!" -ForegroundColor Red
    $errors++
}

# Check 5: Port Availability
Write-Host ""
Write-Host "[5/6] Checking port 8000..." -ForegroundColor Yellow

$portInUse = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "  [WARN] Port 8000 is already in use" -ForegroundColor Yellow
    Write-Host "        You may need to stop the existing server" -ForegroundColor Yellow
    $warnings++
} else {
    Write-Host "  [OK] Port 8000 is available" -ForegroundColor Green
}

# Check 6: Network Connectivity
Write-Host ""
Write-Host "[6/6] Checking network connectivity..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "https://www.google.com" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Host "  [OK] Internet connection available" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] No internet connection (required for Edge-TTS)" -ForegroundColor Yellow
    $warnings++
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "[SUCCESS] ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ready to start the server!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run: python main_websocket.py" -ForegroundColor Cyan
} elseif ($errors -eq 0) {
    Write-Host "[PARTIAL] $warnings WARNING(S) - System can run with limitations" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can start the server, but some features may not work" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run: python main_websocket.py" -ForegroundColor Cyan
} else {
    Write-Host "[FAILED] $errors ERROR(S) and $warnings WARNING(S) found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the errors before starting the server" -ForegroundColor Red
    Write-Host ""
    Write-Host "To install missing packages:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements_complete.txt" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
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

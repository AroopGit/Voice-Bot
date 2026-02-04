@echo off
echo ============================================
echo SwiftShip Voice Bot - Startup Script
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Check if requirements are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo ============================================
echo.

REM Start the server
python main.py

pause

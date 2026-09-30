@echo off
title FaceSync - Web Application
echo ============================================================
echo           FaceSync Enterprise - Web Application
echo ============================================================
echo.

:: Check if .venv exists
if not exist ".\.venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Running setup.bat first...
    call setup.bat
    if %errorlevel% neq 0 exit /b 1
)

echo [*] Starting FaceSync FastAPI Server on port 8000...
echo [*] Access the application in your browser at:
echo     Local:   http://localhost:8000
echo.

:: Open default browser after a 3-second delay in background
start "" /b cmd /c "timeout /t 3 >nul & start http://localhost:8000"

:: Local mode: the BROWSER runs the face models, dev login enabled, frontend served by the API.
:: To override, set the variable in Windows (or edit this file). These take priority over .env.
if not defined INFERENCE_MODE set INFERENCE_MODE=device
if not defined DEMO_MODE set DEMO_MODE=true
if not defined SERVE_FRONTEND set SERVE_FRONTEND=true
if not defined FRONTEND_ORIGINS set FRONTEND_ORIGINS=*

:: Start Uvicorn web server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause

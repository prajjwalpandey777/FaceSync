@echo off
title FaceSync - One-Time Setup
echo ============================================================
echo           FaceSync Enterprise - One-Time Setup
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected:
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [+] Virtual environment created.
) else (
    echo [*] Virtual environment (.venv) already exists.
)

echo.
echo [*] Upgrading pip...
.\.venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo [*] Installing FaceSync dependencies from requirements-local.txt...
echo (This may take a few minutes on the first run...)
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some packages failed to install. Please check your internet connection or Python version.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [+] Setup complete! You are ready to run FaceSync.
echo.
echo - To launch the Web Application, double-click: run_web.bat
echo - To launch the Desktop Webcam App, double-click: run_desktop.bat
echo ============================================================
echo.
pause

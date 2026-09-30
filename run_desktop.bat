@echo off
title FaceSync - Real-Time Webcam Attendance
echo ============================================================
echo      FaceSync - Real-Time Webcam Attendance Application
echo ============================================================
echo.

:: Check if .venv exists
if not exist ".\.venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Running setup.bat first...
    call setup.bat
    if %errorlevel% neq 0 exit /b 1
)

echo [*] Launching Desktop Attendance System...
echo     - Camera Mirror Mode: ON
echo     - MiniFASNetV2 Anti-Spoofing: ON
echo.
echo Controls in Camera Window:
echo   [q] or [ESC] : Exit
echo   [c]         : Clear session attendance
echo   [s]         : Save screenshot
echo.

.\.venv\Scripts\python.exe webcam_attendance.py
pause

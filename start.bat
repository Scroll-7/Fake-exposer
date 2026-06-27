@echo off
title Fake News Detector
cd /d "%~dp0"

:: Check prerequisites
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.x.
    pause
    exit /b 1
)

where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js not found in PATH. Please install Node.js.
    pause
    exit /b 1
)

:: Install npm deps if missing
if not exist "node_modules\" (
    echo [INFO] Installing npm dependencies...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

:: Install Python deps if missing
pip show fastapi >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing Python dependencies...
    pip install -r requirements.txt
)

echo.
echo ==========================================
echo   Fake News Detector - Starting all services
echo ==========================================
echo.
echo   Server will spawn all Python APIs (ports 8000-8002)
echo   Open http://localhost:3001 when ready
echo.
echo   Press Ctrl+C to stop all services
echo ==========================================
echo.

:: Single npm start — server.js handles all Python API spawning
npm start

:: If npm start exits, pause so user can see the error
echo.
echo [Server stopped]
pause

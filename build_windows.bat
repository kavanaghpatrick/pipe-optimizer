@echo off
REM Build script for Pipe Optimizer Windows executable
REM Run this on a Windows machine with Python installed

echo ==========================================
echo Building Pipe Optimizer Windows App
echo ==========================================

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install pyinstaller pandas numpy openpyxl pulp psutil

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build
echo Building executable...
pyinstaller PipeOptimizer.spec --clean --noconfirm

echo.
echo ==========================================
echo BUILD COMPLETE!
echo ==========================================
echo.
echo Your app is at: dist\Pipe Optimizer\PipeOptimizer.exe
echo.
echo To distribute, zip the entire "dist\Pipe Optimizer" folder
echo.
pause

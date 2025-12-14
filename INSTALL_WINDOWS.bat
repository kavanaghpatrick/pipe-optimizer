@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Pipe Optimizer - Windows Installer
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Downloading Python installer...
    echo.

    :: Download Python installer
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python_installer.exe'"

    if not exist python_installer.exe (
        echo ERROR: Failed to download Python installer.
        echo Please install Python 3.11+ manually from python.org
        pause
        exit /b 1
    )

    echo Installing Python (this may take a minute)...
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

    :: Clean up
    del python_installer.exe

    echo Python installed! Please close this window and run the installer again.
    pause
    exit /b 0
)

echo Python found:
python --version
echo.

:: Install required packages
echo Installing required packages...
pip install --upgrade pip
pip install pandas numpy openpyxl pulp psutil

if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To run the Pipe Optimizer:
echo   1. Double-click RUN_OPTIMIZER.bat
echo   2. Or run: python pipe_optimizer_gui.py
echo.

:: Create a run script
echo @echo off > RUN_OPTIMIZER.bat
echo cd /d "%%~dp0" >> RUN_OPTIMIZER.bat
echo python pipe_optimizer_gui.py >> RUN_OPTIMIZER.bat
echo pause >> RUN_OPTIMIZER.bat

echo Created RUN_OPTIMIZER.bat - double-click to run!
echo.
pause

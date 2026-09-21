@echo off
setlocal
title LEGO Konstruktor - rebuild exe
cd /d "%~dp0"

echo Current folder: %cd%
echo.

if not exist "venv\Scripts\python.exe" (
    echo [First run] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10-3.12 from python.org
        echo and check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo [First run] Installing app dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo [First run] Installing PyInstaller...
    python -m pip install pyinstaller
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [1/2] Rebuilding exe from current app.py...

if exist "venv\Scripts\pyinstaller.exe" (
    venv\Scripts\pyinstaller.exe build.spec --clean --noconfirm
) else (
    venv\Scripts\python.exe -m PyInstaller build.spec --clean --noconfirm
)

if errorlevel 1 (
    echo.
    echo BUILD FAILED - see error text above, usually a missing hiddenimport.
    pause
    exit /b 1
)

echo.
echo [2/2] Done! File is at: dist\LEGO_Konstruktor\LEGO_Konstruktor.exe
echo Copy the whole dist\LEGO_Konstruktor folder to move it to another PC.
echo.
pause

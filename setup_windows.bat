@echo off
setlocal

echo ========================================
echo VALOR experiment environment setup
echo ========================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ first.
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv .venv
if errorlevel 1 exit /b 1

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [4/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo ========================================
echo Setup complete.
echo Activate later with:
echo .venv\Scripts\activate.bat
echo ========================================

endlocal

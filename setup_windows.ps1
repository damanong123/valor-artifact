$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "VALOR experiment environment setup"
Write-Host "========================================"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found. Please install Python 3.11+ first."
    exit 1
}

Write-Host "[1/4] Creating virtual environment..."
python -m venv .venv

Write-Host "[2/4] Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "[3/4] Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "[4/4] Installing dependencies..."
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================"
Write-Host "Setup complete."
Write-Host "Activate later with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "========================================"

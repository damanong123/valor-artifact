#!/usr/bin/env bash
set -e

echo "========================================"
echo "VALOR experiment environment setup"
echo "========================================"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Please install Python 3.11+ first."
  exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv .venv

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

echo "[3/4] Upgrading pip..."
python -m pip install --upgrade pip

echo "[4/4] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup complete."
echo "Activate later with:"
echo "source .venv/bin/activate"
echo "========================================"

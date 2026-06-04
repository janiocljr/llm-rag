#!/usr/bin/env bash
set -euo pipefail

echo "Setup: backend (macOS) — Conda or pyenv-based installer"

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BACKEND_REQS="$ROOT_DIR/backend/requirements.txt"

command -v brew >/dev/null 2>&1 || { echo "Homebrew is required. Install it from https://brew.sh and re-run."; exit 1; }

echo "Ensuring Ghostscript is installed (brew)..."
brew install ghostscript || true

ensure_miniforge() {
  if [ -x "$HOME/miniforge3/bin/conda" ]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
    return 0
  fi

  echo "Miniforge not found — installing Miniforge (conda-forge)..."
  curl -L -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
  bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
  export PATH="$HOME/miniforge3/bin:$PATH"
  rm -f /tmp/Miniforge3.sh
}

if command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then
  CONDA_CMD=$(command -v mamba 2>/dev/null || command -v conda)
else
  ensure_miniforge
  CONDA_CMD=$(command -v mamba 2>/dev/null || command -v conda || true)
fi

if [ -n "$CONDA_CMD" ]; then
  echo "Found Conda/Mamba: $CONDA_CMD — creating conda env 'rag' with Python 3.11"
  $CONDA_CMD create -n rag python=3.11 -y -c conda-forge numpy pandas ghostscript pip || true
  echo "Installing remaining Python requirements into 'rag' using conda run..."
  $CONDA_CMD run -n rag pip install --upgrade pip setuptools wheel
  $CONDA_CMD run -n rag pip install -r "$BACKEND_REQS"
  echo "Conda env 'rag' ready. Activate with: conda activate rag"
  exit 0
fi

echo "No Conda found. Falling back to pyenv + venv approach."

command -v pyenv >/dev/null 2>&1 || {
  echo "pyenv not found — installing pyenv via brew"
  brew install pyenv
}

PYTHON_VERSION=3.11.10
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
  echo "Installing Python ${PYTHON_VERSION} via pyenv (this can take a few minutes)..."
  env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install "${PYTHON_VERSION}"
fi

echo "Setting local python to ${PYTHON_VERSION}"
pyenv local "${PYTHON_VERSION}"

echo "Creating virtual environment .venv"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "Installing backend requirements (this may compile some packages)..."
pip install -r "$BACKEND_REQS"

echo "Virtualenv created at .venv — activate with: source .venv/bin/activate"
echo "If install fails building numpy/pandas, consider installing Miniforge/conda and re-run this script."

echo "Done."

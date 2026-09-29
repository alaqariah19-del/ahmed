#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$ROOT/engines"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required."
  exit 1
fi

if [ ! -d "$ROOT/engines/facefusion/.git" ]; then
  git clone --depth 1 https://github.com/facefusion/facefusion.git "$ROOT/engines/facefusion"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not available in this Codespace."
  echo "Install/enable Miniconda, then run this script again."
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --name facefusion python=3.12 pip -y || true
conda activate facefusion
cd "$ROOT/engines/facefusion"

if command -v nvidia-smi >/dev/null 2>&1; then
  python install.py --onnxruntime cuda
else
  python install.py --onnxruntime default
fi

python facefusion.py force-download --download-scope lite --download-providers github huggingface || true

echo "FaceFusion setup finished."

#!/usr/bin/env bash
# Install ATLAS dependencies on a single Colab A100 runtime using two venvs:
# - .venv-train : reward-weighted LoRA training stack (torch 2.6 + flash-attn)
# - .venv-serve : vLLM / OpenEvolve serving + search stack
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PY_TAG="$(python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"

if [[ "$PY_MM" == "3.12" ]]; then
  FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
elif [[ "$PY_MM" == "3.11" ]]; then
  FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
else
  echo "Unsupported Python version for the prepared Colab install: $PY_MM"
  echo "Expected 3.11 or 3.12."
  exit 2
fi

echo "[colab_install] repo root: $ROOT"
echo "[colab_install] python:    $PY_MM ($PY_TAG)"
python -m pip install --upgrade pip setuptools wheel virtualenv

# ---------------------------------------------------------------------------
# Training env
# ---------------------------------------------------------------------------
rm -rf "$ROOT/.venv-train"
python -m virtualenv "$ROOT/.venv-train"
source "$ROOT/.venv-train/bin/activate"
pip install --upgrade pip setuptools wheel
pip install \
  "torch==2.6.0" \
  "transformers>=4.56,<5" \
  "peft>=0.14" \
  "trl>=0.14" \
  "accelerate>=1.2" \
  "datasets>=3.2" \
  "sentencepiece" \
  "safetensors" \
  "pyyaml>=6" \
  "openai>=1.50" \
  "tqdm" \
  "kernels>=0.7" \
  "ninja" \
  "packaging"
pip install --no-deps "$FLASH_ATTN_WHL"
pip install -e .
deactivate

# ---------------------------------------------------------------------------
# Serving/search env
# ---------------------------------------------------------------------------
rm -rf "$ROOT/.venv-serve"
python -m virtualenv "$ROOT/.venv-serve"
source "$ROOT/.venv-serve/bin/activate"
pip install --upgrade pip setuptools wheel
# Let vLLM own its torch pin here; do not preinstall torch.
pip install \
  "vllm==0.19.1" \
  "openevolve==0.2.27" \
  "openai>=1.50" \
  "pyyaml>=6" \
  "tqdm" \
  "numpy" \
  "packaging"
pip install -e .
deactivate

echo
echo "[colab_install] done"
echo "  train python: $ROOT/.venv-train/bin/python"
echo "  serve python: $ROOT/.venv-serve/bin/python"
echo "  wrappers:     setup/run_train_python.sh, setup/run_serve_python.sh"

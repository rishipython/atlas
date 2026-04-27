#!/usr/bin/env bash
# Install ATLAS dependencies on a single Colab A100 runtime.
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

# Core runtime stack for local training / eval / OpenEvolve.
pip install \
  "torch==2.6.0" \
  "transformers>=4.56" \
  "peft>=0.14" \
  "trl>=0.14" \
  "accelerate>=1.2" \
  "datasets>=3.2" \
  "sentencepiece" \
  "safetensors" \
  "openai>=1.50" \
  "pyyaml>=6" \
  "tqdm" \
  "vllm==0.19.1" \
  "openevolve==0.2.27" \
  "kernels>=0.7" \
  "ninja" \
  "packaging"

# Flash-attn wheel matching torch 2.6 / CUDA 12 / Python 3.11 on Colab.
pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install -e .

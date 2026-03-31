#!/usr/bin/env bash
# Install all dependencies for the atlas project.
set -euo pipefail

echo "=== Installing ATLAS dependencies ==="
conda activate atlas

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install vllm openai pyyaml tqdm triton

echo "=== Installing atlas in editable mode ==="
pip install -e .

echo "=== Done ==="

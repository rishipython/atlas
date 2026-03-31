#!/usr/bin/env bash
# Start a vLLM server for ATLAS experiments.
# Usage: bash setup/start_vllm.sh [MODEL] [TP_SIZE] [PORT]
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-8B}"
TP_SIZE="${2:-2}"
PORT="${3:-8000}"
GPU_MEM="${4:-0.90}"

echo "Starting vLLM server"
echo "  model:  $MODEL"
echo "  tp:     $TP_SIZE"
echo "  port:   $PORT"
echo "  gpu_mem: $GPU_MEM"

conda activate atlas

exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code

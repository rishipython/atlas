#!/usr/bin/env bash
set -euo pipefail

# Launch FFT phase ablation training + eval in parallel (detached Modal apps).
#
# Usage:
#   ./setup/run_fft_phase_ablation_and_eval.sh
#
# Optional overrides:
#   MODAL_BIN=/path/to/modal MODAL_PROFILE=rishipython ./setup/run_fft_phase_ablation_and_eval.sh

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"

if [[ ! -x "$MODAL_BIN" ]]; then
  echo "ERROR: modal binary not found: $MODAL_BIN" >&2
  exit 1
fi

run_detached() {
  local name="$1"
  shift
  echo "[launch] $name"
  MODAL_PROFILE="$MODAL_PROFILE" "$MODAL_BIN" run -d "$@"
}

echo "[info] profile: $MODAL_PROFILE"
MODAL_PROFILE="$MODAL_PROFILE" "$MODAL_BIN" profile current || true

# 1) Train phase1-only and phase2-only in parallel
run_detached "train phase1-only" \
  experiment/train_atlas_sft.py \
  --dataset data/sft/fftconv_nodiff_twostage_p1_v1.jsonl \
  --phase phase1 \
  --run-name atlas_fftconv_phase1_only_v1 \
  --epochs 3 \
  --learning-rate 5e-5

run_detached "train phase2-only" \
  experiment/train_atlas_sft.py \
  --dataset data/sft/fftconv_nodiff_twostage_p2_v1.jsonl \
  --phase phase2 \
  --run-name atlas_fftconv_phase2_only_v1 \
  --epochs 3 \
  --learning-rate 5e-5

echo "[info] wait for both training apps to finish before starting eval."
echo "       You can monitor with:"
echo "       MODAL_PROFILE=$MODAL_PROFILE $MODAL_BIN app list"

echo "[next] After training completes, run:"
cat <<'EOF'
MODAL_PROFILE=${MODAL_PROFILE:-rishipython} /Users/rishi/miniconda3/envs/atlas/bin/modal run -d experiment/eval_algotune.py \
  --run-name eval_fft_phase1_only_v1 \
  --problems fft_convolution \
  --n-samples 40 --seed 42 \
  --adapter-name atlas_fftconv_phase1_only_v1 \
  --no-eval-base

MODAL_PROFILE=${MODAL_PROFILE:-rishipython} /Users/rishi/miniconda3/envs/atlas/bin/modal run -d experiment/eval_algotune.py \
  --run-name eval_fft_phase2_only_v1 \
  --problems fft_convolution \
  --n-samples 40 --seed 42 \
  --adapter-name atlas_fftconv_phase2_only_v1 \
  --no-eval-base
EOF

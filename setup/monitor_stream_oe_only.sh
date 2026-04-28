#!/usr/bin/env bash
set -euo pipefail

RUN_PREFIX="${1:-stream_oe5_v1}"
PROBLEMS="${2:-fft_convolution,affine_transform_2d,base64_encoding,sha256_hashing,rotate_2d}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"

echo "[monitor] profile=$MODAL_PROFILE run_prefix=$RUN_PREFIX problems=$PROBLEMS"
echo "[monitor] web monitor: /tmp/atlas_streaming_monitor_${RUN_PREFIX}.html"

MODAL_PROFILE="$MODAL_PROFILE" \
WEB_PATH="/tmp/atlas_streaming_monitor_${RUN_PREFIX}.html" \
./setup/monitor_streaming_benchmark.sh \
  --stream-only \
  --oe-only \
  --run-prefix "$RUN_PREFIX" \
  --problems "$PROBLEMS" \
  --web

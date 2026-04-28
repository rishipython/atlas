#!/usr/bin/env bash
set -euo pipefail
REPO="/Users/rishi/cs288/atlas"
RUN_PREFIX="${RUN_PREFIX:-stream_perm_v1}"
STATE_DIR="$REPO/runs/${RUN_PREFIX}_auto_chain"
for f in supervisor.pid runner.pid; do
  if [[ -f "$STATE_DIR/$f" ]]; then
    pid="$(cat "$STATE_DIR/$f" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]]; then kill "$pid" 2>/dev/null || true; fi
  fi
done
pkill -f "setup/auto_stream_chain_v1.py --run-prefix ${RUN_PREFIX}" || true
rm -rf "$STATE_DIR/.supervisor.lock" || true
echo "stopped"

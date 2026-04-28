#!/usr/bin/env bash
set -euo pipefail

REPO="/Users/rishi/cs288/atlas"
PY="/Users/rishi/miniconda3/envs/atlas/bin/python"
RUN_PREFIX="${RUN_PREFIX:-stream_perm_v1}"
PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS="${POLL_SECS:-30}"

STATE_DIR="$REPO/runs/${RUN_PREFIX}_auto_chain"
mkdir -p "$STATE_DIR"
RUNNER_PID_FILE="$STATE_DIR/runner.pid"
SUP_PID_FILE="$STATE_DIR/supervisor.pid"
SUP_LOG="$STATE_DIR/supervisor.log"
RUN_LOG="$STATE_DIR/runner.log"
LOCKDIR="$STATE_DIR/.supervisor.lock"

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  # Recover from stale lock if recorded supervisor PID is gone.
  if [[ -f "$SUP_PID_FILE" ]]; then
    old_pid="$(cat "$SUP_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "supervisor already running (pid=$old_pid)"
      exit 0
    fi
  fi
  rm -rf "$LOCKDIR" 2>/dev/null || true
  mkdir "$LOCKDIR"
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

echo $$ > "$SUP_PID_FILE"
echo "[$(date '+%F %T')] supervisor started pid=$$" >> "$SUP_LOG"

start_runner() {
  pushd "$REPO" >/dev/null
  nohup "$PY" -u setup/auto_stream_chain_v1.py \
      --run-prefix "$RUN_PREFIX" \
      --modal-profile "$PROFILE" \
      --poll-secs "$POLL_SECS" >> "$RUN_LOG" 2>&1 &
  rpid=$!
  popd >/dev/null
  echo "$rpid" > "$RUNNER_PID_FILE"
  echo "[$(date '+%F %T')] runner started pid=$rpid" >> "$SUP_LOG"
}

is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

while true; do
  pid=""
  if [[ -f "$RUNNER_PID_FILE" ]]; then
    pid="$(cat "$RUNNER_PID_FILE" 2>/dev/null || true)"
  fi
  if ! is_alive "$pid"; then
    echo "[$(date '+%F %T')] runner missing/dead, restarting" >> "$SUP_LOG"
    start_runner
  fi
  # heartbeat for monitor/debug
  if [[ -f "$RUNNER_PID_FILE" ]]; then
    pid="$(cat "$RUNNER_PID_FILE" 2>/dev/null || true)"
  fi
  printf '{"ts": %s, "supervisor_pid": %s, "runner_pid": %s}\n' "$(date +%s)" "$$" "${pid:-0}" > "$STATE_DIR/supervisor_heartbeat.json"
  sleep 20
done

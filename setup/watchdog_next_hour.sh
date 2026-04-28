#!/usr/bin/env bash
set -u

REPO="/Users/rishi/cs288/atlas"
MODAL="/Users/rishi/miniconda3/envs/atlas/bin/modal"
PY="/Users/rishi/miniconda3/envs/atlas/bin/python"
PROFILE="${MODAL_PROFILE:-rishipython}"
END_TS=$(( $(date +%s) + 3600 ))

text() {
  "$PY" "$REPO/send_update_over_text.py" "$1" >/dev/null 2>&1 || true
}

check_once() {
  local now oe synth err msg app_json
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  err=0

  MODAL_PROFILE="$PROFILE" bash "$REPO/setup/monitor_oe_base4.sh" > "$REPO/runs/watchdog_monitor_snapshot.txt" 2> "$REPO/runs/watchdog_monitor_err.txt" || err=1

  app_json="$(MODAL_PROFILE="$PROFILE" "$MODAL" app list --json 2>/dev/null || echo '[]')"
  oe=$(printf '%s' "$app_json" | rg -c '"Description": "atlas-openevolve"' || true)
  synth=$(printf '%s' "$app_json" | rg -c '"Description": "atlas-synth-reasoning"' || true)

  if [[ "$err" -eq 1 ]]; then
    msg="[WATCHDOG] $now monitor check failed; see runs/watchdog_monitor_err.txt"
  else
    msg="[WATCHDOG] $now all good. oe_apps=$oe synth_apps=$synth"
  fi
  text "$msg"
}

text "[WATCHDOG] Starting 1-hour sleep+monitor loop (5-min cadence)."
while [[ $(date +%s) -lt $END_TS ]]; do
  sleep 300
  check_once
done
text "[WATCHDOG] Completed 1-hour monitoring loop."

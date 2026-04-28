#!/usr/bin/env bash
set -euo pipefail

# Live progress monitor for detached Modal runs.
#
# Usage examples:
#   MODAL_PROFILE=rishipython ./setup/monitor_modal_progress.sh
#   MODAL_PROFILE=rishipython ./setup/monitor_modal_progress.sh --filter fft
#   MODAL_PROFILE=rishipython ./setup/monitor_modal_progress.sh --apps ap-123 ap-456

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS=20
FILTER=""
APPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-secs)
      POLL_SECS="${2:?missing value}"
      shift 2
      ;;
    --filter)
      FILTER="${2:?missing value}"
      shift 2
      ;;
    --apps)
      shift
      while [[ $# -gt 0 ]]; do
        [[ "$1" == --* ]] && break
        APPS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$MODAL_BIN" ]]; then
  echo "modal binary not found: $MODAL_BIN" >&2
  exit 1
fi

list_apps_json() {
  MODAL_PROFILE="$MODAL_PROFILE" "$MODAL_BIN" app list --json
}

tail_logs() {
  local app_id="$1"
  MODAL_PROFILE="$MODAL_PROFILE" "$MODAL_BIN" app logs "$app_id" --tail 400 2>/dev/null || true
}

while true; do
  clear
  echo "Modal Progress Monitor  profile=$MODAL_PROFILE  poll=${POLL_SECS}s  time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo

  APPS_CSV=""
  if ((${#APPS[@]} > 0)); then
    APPS_CSV="$(printf '%s,' "${APPS[@]}")"
    APPS_CSV="${APPS_CSV%,}"
  fi

  APPS_CSV="$APPS_CSV" FILTER="$FILTER" MODAL_PROFILE="$MODAL_PROFILE" MODAL_BIN="$MODAL_BIN" \
  /Users/rishi/miniconda3/envs/atlas/bin/python - <<'PY'
import json, os, re, subprocess, sys

modal_bin = os.environ["MODAL_BIN"]
modal_profile = os.environ["MODAL_PROFILE"]
filter_text = os.environ.get("FILTER", "").strip().lower()
apps_csv = os.environ.get("APPS_CSV", "").strip()
explicit_apps = set(a.strip() for a in apps_csv.split(",") if a.strip())

def sh(args):
    env = os.environ.copy()
    env["MODAL_PROFILE"] = modal_profile
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if p.returncode != 0:
        return ""
    return p.stdout

raw = sh([modal_bin, "app", "list", "--json"])
if not raw:
    print("Could not fetch app list.")
    sys.exit(0)

try:
    apps = json.loads(raw)
except Exception:
    print("Could not parse app list JSON.")
    sys.exit(0)

if explicit_apps:
    apps = [a for a in apps if a.get("app_id") in explicit_apps or a.get("description") in explicit_apps]
else:
    apps = [a for a in apps if str(a.get("state","")).upper() in {"RUNNING","DEPLOYED","INITIALIZING","PENDING"}]

if filter_text:
    apps = [a for a in apps if filter_text in (a.get("description","").lower() + " " + a.get("app_id","").lower())]

if not apps:
    print("No matching running apps.")
    sys.exit(0)

def progress_from_logs(logs: str):
    # Supports patterns like "sample 7/40", "iteration 12/40", "epoch 2/3", "step 300/1000".
    pats = [
        r"sample\\s+(\\d+)\\s*/\\s*(\\d+)",
        r"iteration\\s+(\\d+)\\s*/\\s*(\\d+)",
        r"iter\\s+(\\d+)\\s*/\\s*(\\d+)",
        r"epoch\\s+(\\d+)\\s*/\\s*(\\d+)",
        r"step\\s+(\\d+)\\s*/\\s*(\\d+)",
    ]
    best = None
    for pat in pats:
        for m in re.finditer(pat, logs, flags=re.IGNORECASE):
            cur, tot = int(m.group(1)), int(m.group(2))
            if tot <= 0:
                continue
            pct = max(0.0, min(100.0, 100.0 * cur / tot))
            best = (cur, tot, pct, m.group(0))
    return best

def bar(pct: float, w: int = 20):
    done = int((pct / 100.0) * w)
    return "[" + "#" * done + "-" * (w - done) + "]"

print(f"{'APP_ID':<14} {'STATE':<12} {'PROGRESS':<34} {'LAST_SIGNAL'}")
print("-" * 100)
for a in apps:
    app_id = a.get("app_id", "")
    state = str(a.get("state", ""))
    logs = sh([modal_bin, "app", "logs", app_id, "--tail", "400"])
    p = progress_from_logs(logs)
    if p:
        cur, tot, pct, sig = p
        prog = f"{bar(pct)} {pct:5.1f}% ({cur}/{tot})"
        last_sig = sig
    else:
        prog = f"{bar(0)}   n/a"
        lines = [ln for ln in logs.splitlines() if ln.strip()]
        last_sig = (lines[-1][:55] + "...") if lines else "no logs yet"
    print(f"{app_id:<14} {state:<12} {prog:<34} {last_sig}")
PY

  echo
  echo "Tip: for full logs of one app:"
  echo "  MODAL_PROFILE=$MODAL_PROFILE $MODAL_BIN app logs <APP_ID> -f --timestamps"
  sleep "$POLL_SECS"
done


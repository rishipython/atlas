#!/usr/bin/env bash
set -euo pipefail

# Focused monitor for convolve2d pass@40 eval runs.
# Shows per-run progress and best correct speedup from summary.partial.json.
#
# Usage:
#   MODAL_PROFILE=rishipython ./setup/monitor_conv2d_eval_table.sh
#   MODAL_PROFILE=rishipython ./setup/monitor_conv2d_eval_table.sh --once
#   MODAL_PROFILE=rishipython ./setup/monitor_conv2d_eval_table.sh --poll-secs 30

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS=20
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-secs)
      POLL_SECS="${2:?missing value}"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
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

RUNS_JSON='[
  {"label":"base","run":"conv2d_base_pass40_h30_v1","leg":"base"},
  {"label":"bestspeed","run":"conv2d_bestspeed_pass40_h30_v1","leg":"atlas"},
  {"label":"twostage_dpo_p2","run":"conv2d_twostage_dpop2_pass40_h30_v1","leg":"atlas"}
]'

while true; do
  clear || true
  echo "Convolve2D Eval Monitor  profile=$MODAL_PROFILE  poll=${POLL_SECS}s  time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo
  MODAL_PROFILE="$MODAL_PROFILE" MODAL_BIN="$MODAL_BIN" RUNS_JSON="$RUNS_JSON" \
  /Users/rishi/miniconda3/envs/atlas/bin/python - <<'PY'
import json
import os
import pathlib
import tempfile
import subprocess

modal_bin = os.environ["MODAL_BIN"]
modal_profile = os.environ["MODAL_PROFILE"]
runs = json.loads(os.environ["RUNS_JSON"])

def sh(args):
    env = os.environ.copy()
    env["MODAL_PROFILE"] = modal_profile
    return subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )

def fetch_summary(run_name: str, leg: str):
    base = f"/eval/{run_name}/convolve2d_full_fill__{leg}"
    for fname in ("summary.json", "summary.partial.json"):
        remote = f"{base}/{fname}"
        with tempfile.TemporaryDirectory(prefix="conv2d_mon_") as td:
            local = str(pathlib.Path(td) / fname)
            p = sh([modal_bin, "volume", "get", "atlas-openevolve-outputs", remote, local])
            if p.returncode == 0 and pathlib.Path(local).exists():
                try:
                    return json.loads(pathlib.Path(local).read_text())
                except Exception:
                    return None
    return None

print(f"{'run':<18} {'progress':<12} {'correct':<8} {'pass@k':<8} {'best_speedup':<12} {'status':<12}")
print("-" * 82)
for r in runs:
    d = fetch_summary(r["run"], r["leg"])
    if not d:
        print(f"{r['label']:<18} {'n/a':<12} {'n/a':<8} {'n/a':<8} {'n/a':<12} {'starting':<12}")
        continue
    n = int(d.get("n_samples", 40) or 40)
    c = int(d.get("completed_samples", n) or n)
    nc = int(d.get("num_correct", 0) or 0)
    pk = float(d.get("pass_at_k", 0.0) or 0.0)
    bs = float(d.get("best_speedup_when_correct", 0.0) or 0.0)
    status = "done" if c >= n else "running"
    print(f"{r['label']:<18} {f'{c}/{n}':<12} {nc:<8d} {pk:<8.3f} {bs:<12.4f} {status:<12}")
PY
  echo
  echo "One-run logs:"
  echo "  MODAL_PROFILE=$MODAL_PROFILE $MODAL_BIN app logs <APP_ID> -f --timestamps"
  if [[ "$ONCE" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_SECS"
done

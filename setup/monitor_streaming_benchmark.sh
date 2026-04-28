#!/usr/bin/env bash
set -euo pipefail

# Streaming benchmark monitor:
# - Eval summaries under /eval/*
# - OE progress under /*_oe/oe/evolution_trace.jsonl
#
# Usage:
#   MODAL_PROFILE=rishipython ./setup/monitor_streaming_benchmark.sh
#   MODAL_PROFILE=rishipython ./setup/monitor_streaming_benchmark.sh --once
#   MODAL_PROFILE=rishipython ./setup/monitor_streaming_benchmark.sh --all
#   MODAL_PROFILE=rishipython ./setup/monitor_streaming_benchmark.sh --web

MODAL_BIN="${MODAL_BIN:-/Users/rishi/miniconda3/envs/atlas/bin/modal}"
MODAL_PROFILE="${MODAL_PROFILE:-rishipython}"
POLL_SECS=30
ONCE=0
STREAM_ONLY=1
OE_ONLY=0
WEB=0
WEB_PATH="${WEB_PATH:-/tmp/atlas_streaming_monitor.html}"
OPENED_WEB=0
RUN_PREFIX=""
INCLUDE_RUNS=""
PROBLEMS_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-secs) POLL_SECS="${2:?missing value}"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --stream-only) STREAM_ONLY=1; shift ;;
    --oe-only) OE_ONLY=1; shift ;;
    --all) STREAM_ONLY=0; shift ;;
    --web) WEB=1; shift ;;
    --web-path) WEB_PATH="${2:?missing value}"; shift 2 ;;
    --run-prefix) RUN_PREFIX="${2:?missing value}"; shift 2 ;;
    --include-runs) INCLUDE_RUNS="${2:?missing value}"; shift 2 ;;
    --problems) PROBLEMS_FILTER="${2:?missing value}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

while true; do
  SNAPSHOT="$(
    MODAL_PROFILE="$MODAL_PROFILE" MODAL_BIN="$MODAL_BIN" STREAM_ONLY="$STREAM_ONLY" OE_ONLY="$OE_ONLY" WEB_PATH="$WEB_PATH" RUN_PREFIX="$RUN_PREFIX" INCLUDE_RUNS="$INCLUDE_RUNS" PROBLEMS_FILTER="$PROBLEMS_FILTER" \
    /Users/rishi/miniconda3/envs/atlas/bin/python - <<'PY'
import json
import os
import pathlib
import re
import subprocess
import tempfile
from statistics import mean
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from html import escape

modal_bin = os.environ["MODAL_BIN"]
modal_profile = os.environ["MODAL_PROFILE"]
stream_only = os.environ.get("STREAM_ONLY", "1") == "1"
oe_only = os.environ.get("OE_ONLY", "0") == "1"
web_path = os.environ.get("WEB_PATH", "").strip()
run_prefix = os.environ.get("RUN_PREFIX", "").strip()
run_prefixes = [x.strip() for x in run_prefix.split(",") if x.strip()]
include_runs = {x.strip() for x in os.environ.get("INCLUDE_RUNS", "").split(",") if x.strip()}
problems_filter_raw = os.environ.get("PROBLEMS_FILTER", "").strip()
problems_filter = {x.strip() for x in problems_filter_raw.split(",") if x.strip()}
known_problems = [
    "base64_encoding",
    "sha256_hashing",
    "rotate_2d",
    "convolve_1d",
    "correlate_1d",
    "dct_type_I_scipy_fftpack",
    "dst_type_II_scipy_fftpack",
    "fft_real_scipy_fftpack",
    "matrix_multiplication",
    "outer_product",
    "shift_2d",
    "fft_cmplx_scipy_fftpack",
    "lu_factorization",
    "polynomial_real",
    "fft_convolution",
    "affine_transform_2d",
    "eigenvectors_complex",
    "psd_cone_projection",
    "convolve2d_full_fill",
]

def sh(args):
    env = os.environ.copy()
    env["MODAL_PROFILE"] = modal_profile
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

def vol_ls(path):
    p = sh([modal_bin, "volume", "ls", "atlas-openevolve-outputs", path])
    if p.returncode != 0:
        return []
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]

def vol_get(remote_path):
    with tempfile.TemporaryDirectory(prefix="stream_mon_") as td:
        local = pathlib.Path(td) / "x.txt"
        p = sh([modal_bin, "volume", "get", "atlas-openevolve-outputs", remote_path, str(local)])
        if p.returncode != 0 or not local.exists():
            return None
        return local.read_text()

def vol_ls_models(path):
    p = sh([modal_bin, "volume", "ls", "atlas-models", path])
    if p.returncode != 0:
        return []
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]

def infer_model(name: str) -> str:
    n = name.lower()
    if "rlm" in n:
        return "rlm"
    if "bestspeed" in n or "besttraj" in n or "best_traj" in n:
        return "sft-best-traj"
    if "dpo" in n or "twostage" in n or "weighted" in n:
        return "dpo-weighted-sft"
    if "base" in n:
        return "base"
    return "unknown"

def infer_problem(name: str) -> str:
    n = name.lower()
    aliases = {
        "fft_cmplx": "fft_cmplx_scipy_fftpack",
        "fftcmplx": "fft_cmplx_scipy_fftpack",
        "affine": "affine_transform_2d",
        "convolve2d": "convolve2d_full_fill",
    }
    for p in known_problems:
        if p in n:
            return p
    for k, v in aliases.items():
        if k in n:
            return v
    return "unknown"

def is_correct_from_metrics(metrics: dict) -> bool:
    # Primary
    if "correctness" in metrics:
        try:
            return float(metrics.get("correctness", 0.0) or 0.0) >= 0.99
        except Exception:
            pass
    # Common fallback in AlgoTune evaluators
    if "correctness_score" in metrics:
        try:
            return float(metrics.get("correctness_score", 0.0) or 0.0) >= 0.99
        except Exception:
            pass
    # Last-resort fallback
    rs = float(metrics.get("runs_successfully", 0.0) or 0.0)
    bf = float(metrics.get("basic_functionality", 0.0) or 0.0)
    return rs >= 0.99 and bf >= 0.99

def speedup_from_metrics(metrics: dict) -> float:
    for k in ("speedup", "speedup_score", "score"):
        if k in metrics:
            try:
                return float(metrics.get(k, 0.0) or 0.0)
            except Exception:
                pass
    return 0.0

rows = []
now_pst = datetime.now(ZoneInfo("America/Los_Angeles"))

def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds <= 0:
        return "0s"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"

def fmt_finish(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    t = now_pst + timedelta(seconds=max(0.0, seconds))
    return t.strftime("%a %Y-%m-%d %H:%M:%S %Z")

# Distillation rows for fft-conv adapters used by atlas methods.
distill_specs = [
    {"model": "sft-best-traj", "run": "atlas_fftconv_bestspeed_only_v2"},
    {"model": "dpo-weighted-sft", "run": "atlas_fftconv_dpo_phase1_v2"},
]
distill_state_by_model = {}
for spec in distill_specs:
    model = spec["model"]
    run = spec["run"]
    entries = vol_ls_models(f"/{run}")
    if not entries:
        st = {
            "model": model,
            "method": "distill",
            "problem": "fft_convolution",
            "progress": "n/a",
            "correct": "n/a",
            "passk": "n/a",
            "best": "n/a",
            "mean": "n/a",
            "status": "waiting",
            "avg_t": "n/a",
            "last_t": "n/a",
        }
        rows.append(st)
        distill_state_by_model[model] = st
        continue
    summary = vol_get(f"/{run}/training_summary.json")
    if summary:
        st = {
            "model": model,
            "method": "distill",
            "problem": "fft_convolution",
            "progress": "done",
            "correct": "n/a",
            "passk": "n/a",
            "best": "n/a",
            "mean": "n/a",
            "status": "done",
            "avg_t": "n/a",
            "last_t": "n/a",
            "eta_s": 0.0,
        }
        rows.append(st)
        distill_state_by_model[model] = st
        continue
    ckpts = []
    for e in entries:
        m = re.search(rf"{re.escape(run)}/checkpoint-(\d+)$", e)
        if m:
            ckpts.append(int(m.group(1)))
    if not ckpts:
        st = {
            "model": model,
            "method": "distill",
            "problem": "fft_convolution",
            "progress": "starting",
            "correct": "n/a",
            "passk": "n/a",
            "best": "n/a",
            "mean": "n/a",
            "status": "starting",
            "avg_t": "n/a",
            "last_t": "n/a",
        }
        rows.append(st)
        distill_state_by_model[model] = st
        continue
    last_ckpt = max(ckpts)
    ts = vol_get(f"/{run}/checkpoint-{last_ckpt}/trainer_state.json")
    step = last_ckpt
    max_steps = None
    if ts:
        try:
            d = json.loads(ts)
            step = int(d.get("global_step", step) or step)
            ms = d.get("max_steps")
            max_steps = int(ms) if ms is not None else None
        except Exception:
            pass
    prog = f"{step}/{max_steps}" if max_steps else str(step)
    eta = None
    if max_steps and max_steps > step:
        eta = None
    st = {
        "model": model,
        "method": "distill",
        "problem": "fft_convolution",
        "progress": prog,
        "correct": "n/a",
        "passk": "n/a",
        "best": "n/a",
        "mean": "n/a",
        "status": "training",
        "avg_t": "n/a",
        "last_t": "n/a",
        "eta_s": eta,
    }
    rows.append(st)
    distill_state_by_model[model] = st

# OE rows
root_entries = vol_ls("/")
oe_runs = sorted([x for x in root_entries if x.endswith("_oe") and not x.startswith("eval/")])
if stream_only:
    oe_runs = [x for x in oe_runs if x.startswith("stream_")]
if include_runs:
    oe_runs = [x for x in oe_runs if x in include_runs]
elif run_prefixes:
    oe_runs = [x for x in oe_runs if any(x.startswith(p) for p in run_prefixes)]
for run in oe_runs:
    trace_path = f"/{run}/oe/evolution_trace.jsonl"
    txt = vol_get(trace_path)
    if txt is None:
        rows.append({
            "model": infer_model(run),
            "method": "OE",
            "problem": infer_problem(run),
            "progress": "n/a",
            "correct": "n/a",
            "passk": "n/a",
            "best": "n/a",
            "mean": "n/a",
            "status": "starting",
            "avg_t": "n/a",
            "last_t": "n/a",
        })
        continue
    recs = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            recs.append(json.loads(ln))
        except Exception:
            pass
    iters = len(recs)
    correct = []
    iter_times = []
    for r in recs:
        m = r.get("child_metrics") or {}
        s = speedup_from_metrics(m)
        if is_correct_from_metrics(m):
            correct.append(s)
        mt = (r.get("metadata") or {}).get("iteration_time")
        if isinstance(mt, (int, float)):
            iter_times.append(float(mt))
    rows.append({
        "model": infer_model(run),
        "method": "OE",
        "problem": infer_problem(run),
        "progress": f"{iters}/40",
        "correct": str(len(correct)),
        "passk": f"{(len(correct)/iters if iters else 0.0):.3f}",
        "best": f"{(max(correct) if correct else 0.0):.4f}",
        "mean": f"{(mean(correct) if correct else 0.0):.4f}",
        "status": "done" if iters >= 40 else "running",
        "avg_t": f"{(mean(iter_times) if iter_times else 0.0):.2f}s" if iter_times else "n/a",
        "last_t": f"{iter_times[-1]:.2f}s" if iter_times else "n/a",
        "eta_s": (max(0, 40 - iters) * mean(iter_times)) if iter_times and iters < 40 else (0.0 if iters >= 40 else None),
    })

# Atlas model rows should clearly reflect whether they are in training or running OE.
for r in rows:
    if r.get("method") != "OE":
        continue
    model = r.get("model")
    if model in ("sft-best-traj", "dpo-weighted-sft"):
        d = distill_state_by_model.get(model)
        if d and d.get("status") == "training" and r.get("status") in ("waiting", "starting"):
            r["status"] = "training"
        elif r.get("status") == "running":
            r["status"] = "running_oe"
        elif r.get("status") == "done":
            r["status"] = "done_oe"

if not oe_only:
    # Best-of-n eval rows
    eval_entries = vol_ls("/eval")
    eval_runs = sorted([x.split("/", 1)[1] for x in eval_entries if x.startswith("eval/")])
    if stream_only:
        eval_runs = [x for x in eval_runs if x.startswith("stream_")]
    if include_runs:
        eval_runs = [x for x in eval_runs if x in include_runs]
    elif run_prefixes:
        eval_runs = [x for x in eval_runs if any(x.startswith(p) for p in run_prefixes)]
    for run in eval_runs:
        leg_entries = vol_ls(f"/eval/{run}")
        legs = [x.split("/", 2)[2] for x in leg_entries if x.startswith(f"eval/{run}/") and "__" in x]
        for leg in sorted(set(legs)):
            d = None
            for fname in ("summary.json", "summary.partial.json"):
                txt = vol_get(f"/eval/{run}/{leg}/{fname}")
                if txt:
                    try:
                        d = json.loads(txt)
                    except Exception:
                        d = None
                    if d:
                        break
            samples = None
            for fname in ("samples.json", "samples.partial.json"):
                txt = vol_get(f"/eval/{run}/{leg}/{fname}")
                if txt:
                    try:
                        samples = json.loads(txt)
                    except Exception:
                        samples = None
                    if isinstance(samples, list):
                        break
            model = infer_model(run)
            problem = leg.split("__", 1)[0]
            if not d:
                rows.append({
                    "model": model, "method": "best-of-n", "problem": problem,
                    "progress": "n/a", "correct": "n/a", "passk": "n/a",
                    "best": "n/a", "mean": "n/a", "status": "starting",
                    "avg_t": "n/a", "last_t": "n/a",
                })
                continue
            n = int(d.get("n_samples", 40) or 40)
            c = int(d.get("completed_samples", n) or n)
            nc = int(d.get("num_correct", 0) or 0)
            pk = float(d.get("pass_at_k", 0.0) or 0.0)
            bs = float(d.get("best_speedup_when_correct", 0.0) or 0.0)
            ms = float(d.get("mean_speedup_when_correct", 0.0) or 0.0)
            score_times = []
            if isinstance(samples, list):
                for s in samples:
                    t = s.get("score_time_s")
                    if isinstance(t, (int, float)):
                        score_times.append(float(t))
            rows.append({
                "model": model,
                "method": "best-of-n",
                "problem": problem,
                "progress": f"{c}/{n}",
                "correct": str(nc),
                "passk": f"{pk:.3f}",
                "best": f"{bs:.4f}",
                "mean": f"{ms:.4f}",
                "status": "done" if c >= n else "running",
                "avg_t": f"{(mean(score_times) if score_times else 0.0):.2f}s" if score_times else "n/a",
                "last_t": f"{score_times[-1]:.2f}s" if score_times else "n/a",
                "eta_s": (max(0, n - c) * mean(score_times)) if score_times and c < n else (0.0 if c >= n else None),
            })

rows.sort(key=lambda r: (r["model"], r["method"], r["problem"]))

# Narrow to requested problems first (if provided).
if problems_filter:
    rows = [r for r in rows if r.get("problem") in problems_filter]

# Ensure OE streaming grid always renders:
#   model in {base, sft-best-traj, dpo-weighted-sft, rlm}
#   problem in requested filter OR any problem already seen for this run-prefix
expected_models = ["base", "sft-best-traj", "dpo-weighted-sft", "rlm"]
seen_problems = {r.get("problem") for r in rows if r.get("problem") and r.get("problem") != "unknown"}
grid_problems = set(problems_filter) if problems_filter else set()
grid_problems |= seen_problems

existing_oe = {
    (r.get("model"), r.get("problem"))
    for r in rows
    if r.get("method") == "OE"
}
for p in sorted(grid_problems):
    for m in expected_models:
        if (m, p) in existing_oe:
            continue
        rows.append(
            {
                "model": m,
                "method": "OE",
                "problem": p,
                "progress": "n/a",
                "correct": "n/a",
                "passk": "n/a",
                "best": "n/a",
                "mean": "n/a",
                "status": "waiting",
                "avg_t": "n/a",
                "last_t": "n/a",
            }
        )

rows.sort(key=lambda r: (r["model"], r["method"], r["problem"]))
print("model | method | problem | progress | correct | pass@k | best_speedup | mean_speedup | status | avg_time/iter | latest_time/iter | eta_from_now | eta_finish_pst")
print("-" * 186)
if not rows:
    print("(no matching runs)")
for r in rows:
    print(
        f"{r['model']:<16} {r['method']:<10} {r['problem']:<26} {r['progress']:<9} "
        f"{r['correct']:<7} {r['passk']:<7} {r['best']:<12} {r['mean']:<12} {r['status']:<8} "
        f"{r['avg_t']:<13} {r['last_t']:<13} {fmt_eta(r.get('eta_s')):<12} {fmt_finish(r.get('eta_s')):<24}"
    )

if web_path:
    css = """
    :root { --bg:#0b1020; --card:#121936; --text:#e9eef9; --muted:#9fb0d5; --ok:#2ecc71; --run:#f4c542; --bad:#ff7b7b; --line:#28345c; }
    body { margin:0; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:linear-gradient(120deg,#0b1020,#101735); color:var(--text);}
    .wrap { max-width: 1500px; margin: 24px auto; padding: 0 16px; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    .meta { color: var(--muted); margin-bottom: 14px; }
    .card { background: var(--card); border:1px solid var(--line); border-radius: 14px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,.28);}
    table { width:100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { background: rgba(255,255,255,.03); color: #d7e3ff; font-weight: 600; position: sticky; top: 0; }
    tr:hover td { background: rgba(255,255,255,.03); }
    .status-done { color: var(--ok); font-weight: 700; }
    .status-running { color: var(--run); font-weight: 700; }
    .status-starting { color: var(--bad); font-weight: 700; }
    .status-training { color: #9ad0ff; font-weight: 700; }
    .status-running_oe { color: var(--run); font-weight: 700; }
    .status-done_oe { color: var(--ok); font-weight: 700; }
    """
    head = (
        "<tr>"
        "<th>model</th><th>method</th><th>problem</th><th>progress</th><th>correct</th>"
        "<th>pass@k</th><th>best_speedup</th><th>mean_speedup</th><th>status</th>"
        "<th>avg_time/iter</th><th>latest_time/iter</th><th>eta_from_now</th><th>eta_finish_pst</th>"
        "</tr>"
    )
    body = []
    for r in rows:
        st = r["status"]
        cls = f"status-{st}"
        body.append(
            "<tr>"
            f"<td>{escape(str(r['model']))}</td>"
            f"<td>{escape(str(r['method']))}</td>"
            f"<td>{escape(str(r['problem']))}</td>"
            f"<td>{escape(str(r['progress']))}</td>"
            f"<td>{escape(str(r['correct']))}</td>"
            f"<td>{escape(str(r['passk']))}</td>"
            f"<td>{escape(str(r['best']))}</td>"
            f"<td>{escape(str(r['mean']))}</td>"
            f"<td class='{cls}'>{escape(str(st))}</td>"
            f"<td>{escape(str(r['avg_t']))}</td>"
            f"<td>{escape(str(r['last_t']))}</td>"
            f"<td>{escape(fmt_eta(r.get('eta_s')))}</td>"
            f"<td>{escape(fmt_finish(r.get('eta_s')))}</td>"
            "</tr>"
        )
    empty_row = '<tr><td colspan="13">(no matching runs)</td></tr>'
    rows_html = ''.join(body) if body else empty_row
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>ATLAS Streaming Monitor</title>
<style>{css}</style></head>
<body><div class="wrap"><h1>ATLAS Streaming Monitor</h1>
<div class="meta">Updated: {escape(now_pst.strftime('%a %Y-%m-%d %H:%M:%S %Z'))} | Mode: {'oe-only' if oe_only else ('stream-only' if stream_only else 'all-runs')}</div>
<div class="card"><table>{head}{rows_html}</table></div>
</div></body></html>"""
    pathlib.Path(web_path).write_text(html_doc)
PY
  )"
  clear || true
  mode="stream-only"
  if [[ "$OE_ONLY" -eq 1 ]]; then
    mode="oe-only"
  elif [[ "$STREAM_ONLY" -eq 0 ]]; then
    mode="all-runs"
  fi
  echo "Streaming Monitor  profile=$MODAL_PROFILE  mode=$mode  poll=${POLL_SECS}s  time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo
  echo "$SNAPSHOT"
  if [[ "$WEB" -eq 1 && "$OPENED_WEB" -eq 0 ]]; then
    if command -v open >/dev/null 2>&1; then
      open "$WEB_PATH" >/dev/null 2>&1 || true
      OPENED_WEB=1
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$WEB_PATH" >/dev/null 2>&1 || true
      OPENED_WEB=1
    fi
  fi
  echo
  echo "Tip:"
  echo "  MODAL_PROFILE=$MODAL_PROFILE $MODAL_BIN app list --json"
  echo "  MODAL_PROFILE=$MODAL_PROFILE $MODAL_BIN app logs <APP_ID> -f --timestamps"
  if [[ "$WEB" -eq 1 ]]; then
    echo "  Web dashboard: $WEB_PATH"
  fi
  if [[ "$ONCE" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_SECS"
done

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODAL = "/Users/rishi/miniconda3/envs/atlas/bin/modal"


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_trace_count(env: dict[str, str], run_name: str) -> int:
    local = Path("/tmp") / f"{run_name}_trace.jsonl"
    if local.exists():
        local.unlink()
    p = run([MODAL, "volume", "get", "atlas-openevolve-outputs", f"/{run_name}/oe/evolution_trace.jsonl", str(local)], env)
    if p.returncode != 0 or not local.exists():
        return 0
    n = 0
    for ln in local.read_text().splitlines():
        if ln.strip():
            n += 1
    return n


def launch_oe(env: dict[str, str], run_name: str, problem: str, seed: int, iterations: int, adapter: str | None, rlm_bank: str | None) -> None:
    cmd = [
        MODAL,
        "run",
        "-d",
        "experiment/openevolve_runner.py",
        "--problem-id",
        problem,
        "--task-family",
        "algotune",
        "--iterations",
        str(iterations),
        "--random-seed",
        str(seed),
        "--run-name",
        run_name,
    ]
    if adapter:
        cmd += ["--adapter-name", adapter]
    if rlm_bank:
        cmd += ["--rlm-memory-bank", rlm_bank]
    p = run(cmd, env)
    if p.returncode != 0:
        raise RuntimeError(f"launch failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="stream_perm_v1")
    ap.add_argument("--permutation-file", default="setup/stream_permutation_v1.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--poll-secs", type=int, default=60)
    ap.add_argument("--modal-profile", default=os.environ.get("MODAL_PROFILE", "rishipython"))
    args = ap.parse_args()

    env = os.environ.copy()
    env["MODAL_PROFILE"] = args.modal_profile

    problems = [x.strip() for x in Path(args.permutation_file).read_text().splitlines() if x.strip()]
    if len(problems) < 2:
        raise ValueError("Need at least 2 problems in permutation")
    eval_probs = problems[1:]

    methods = {
        "sft-best-traj": {"adapter": "atlas_fftconv_bestspeed_only_v4", "rlm": None},
        "dpo-weighted-sft": {"adapter": "atlas_fftconv_dpo_p2bridge_v4", "rlm": None},
        "rlm": {"adapter": None, "rlm": "data/rlm_memory_fftconv_synth_v1.txt"},
    }

    state_path = REPO / "runs" / f"{args.run_prefix}_auto" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"launched": {}, "completed": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())

    while True:
        changed = False
        for method, cfg in methods.items():
            launched = set(state["launched"].get(method, []))
            completed = set(state["completed"].get(method, []))
            for i, prob in enumerate(eval_probs):
                run_name = f"{args.run_prefix}_{method}_{prob}_oe"
                if prob in completed:
                    continue
                if prob not in launched:
                    # only launch first missing, and only if previous problem completed
                    if i > 0 and eval_probs[i - 1] not in completed:
                        break
                    launch_oe(env, run_name, prob, args.seed, args.iterations, cfg["adapter"], cfg["rlm"])
                    launched.add(prob)
                    state["launched"][method] = sorted(launched)
                    changed = True
                    break
                n = get_trace_count(env, run_name)
                if n >= args.iterations:
                    completed.add(prob)
                    state["completed"][method] = sorted(completed)
                    changed = True
                    continue
                break
        if changed:
            state_path.write_text(json.dumps(state, indent=2))
        all_done = all(len(set(state["completed"].get(m, []))) == len(eval_probs) for m in methods)
        if all_done:
            return
        time.sleep(args.poll_secs)


if __name__ == "__main__":
    main()


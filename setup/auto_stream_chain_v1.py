#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import threading
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODAL = "/Users/rishi/miniconda3/envs/atlas/bin/modal"
PYTHON = "/Users/rishi/miniconda3/envs/atlas/bin/python"


def run(cmd: list[str], env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=str(REPO), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def count_remote_iter_files(env: dict[str, str], remote_dir: str) -> int:
    p = run([MODAL, "volume", "ls", "atlas-openevolve-outputs", remote_dir], env, check=False)
    if p.returncode != 0:
        return 0
    n = 0
    for ln in p.stdout.splitlines():
        if re.search(r"iter_\d+_content\.txt$", ln.strip()):
            n += 1
    return n


def vol_exists(env: dict[str, str], volume: str, remote_path: str) -> bool:
    p = run([MODAL, "volume", "ls", volume, remote_path], env, check=False)
    return p.returncode == 0


def get_trace_count(env: dict[str, str], run_name: str) -> int:
    with tempfile.TemporaryDirectory(prefix="stream_chain_") as td:
        local = Path(td) / "trace.jsonl"
        p = run([MODAL, "volume", "get", "atlas-openevolve-outputs", f"/{run_name}/oe/evolution_trace.jsonl", str(local)], env, check=False)
        if p.returncode != 0 or not local.exists():
            return 0
        return sum(1 for ln in local.read_text().splitlines() if ln.strip())


def ensure_oe_running(
    env: dict[str, str],
    run_name: str,
    problem: str,
    seed: int,
    iterations: int,
    adapter_name: str | None,
    rlm_bank_local: str | None,
) -> None:
    if vol_exists(env, "atlas-openevolve-outputs", f"/{run_name}/oe/evolution_trace.jsonl"):
        return
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
    if adapter_name:
        cmd += ["--adapter-name", adapter_name]
    if rlm_bank_local:
        cmd += ["--rlm-memory-bank", rlm_bank_local]
    run(cmd, env, check=True)


def build_method_update(
    env: dict[str, str],
    run_prefix: str,
    method: str,
    problem: str,
    next_problem: str | None,
    adapter_before: str | None,
    rlm_before: str | None,
) -> tuple[str | None, str | None]:
    local_dir = REPO / "runs" / f"{run_prefix}_auto_chain" / method / problem
    local_dir.mkdir(parents=True, exist_ok=True)
    trace = local_dir / "evolution_trace.jsonl"
    synth_dir = local_dir / "synth"
    p1 = local_dir / "p1.jsonl"
    p2 = local_dir / "p2.jsonl"
    best = local_dir / "best.jsonl"

    run_name = f"{run_prefix}_{method}_{problem}_oe"
    run([MODAL, "volume", "get", "atlas-openevolve-outputs", f"/{run_name}/oe/evolution_trace.jsonl", str(trace), "--force"], env, check=True)

    synth_name = f"{run_prefix}_{method}_{problem}_synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Reuse existing synth trajectories and only generate missing iterations.
    recs = [json.loads(ln) for ln in trace.read_text().splitlines() if ln.strip()]
    rec_by_iter = {int(r.get("iteration", -1)): r for r in recs if int(r.get("iteration", -1)) >= 0}
    all_iters = sorted(rec_by_iter.keys())
    done_iters = set()
    for pth in synth_dir.glob("iter_*_content.txt"):
        m = re.match(r"iter_(\d+)_content\.txt$", pth.name)
        if m:
            done_iters.add(int(m.group(1)))
    missing = [it for it in all_iters if it not in done_iters]

    if missing:
        shard_cfg = {"sft-best-traj": 4, "dpo-weighted-sft": 3, "rlm": 3}
        n_shards = shard_cfg.get(method, 3)
        shards: list[list[int]] = [[] for _ in range(n_shards)]
        for j, it in enumerate(missing):
            shards[j % n_shards].append(it)

        shard_root = local_dir / "synth_shards"
        shard_root.mkdir(parents=True, exist_ok=True)

        def run_one_shard(idx: int, iters: list[int]) -> None:
            if not iters:
                return
            trace_part = shard_root / f"trace_shard_{idx}.jsonl"
            with trace_part.open("w") as f:
                for it in iters:
                    f.write(json.dumps(rec_by_iter[it]) + "\n")
            out_name = f"{synth_name}_shard{idx}"
            p = run(
                [
                    MODAL,
                    "run",
                    "-d",
                    "experiment/synth_reasoning.py::main_algotune",
                    "--trace-path",
                    str(trace_part),
                    "--problem-id",
                    problem,
                    "--out-name",
                    out_name,
                ],
                env,
                check=False,
            )
            if p.returncode != 0:
                raise RuntimeError(f"synth shard {idx} failed:\n{p.stdout}\n{p.stderr}")

            # Detached synth: wait until all expected shard outputs are materialized.
            remote_out = f"/synth/{out_name}"
            deadline = time.time() + 60 * 45
            while time.time() < deadline:
                n_done = count_remote_iter_files(env, remote_out)
                if n_done >= len(iters):
                    break
                time.sleep(20)
            else:
                raise RuntimeError(f"synth shard {idx} timed out waiting for outputs in {remote_out}")

            shard_out = shard_root / f"out_shard_{idx}"
            g = run(
                [MODAL, "volume", "get", "atlas-openevolve-outputs", remote_out, str(shard_out), "--force"],
                env,
                check=False,
            )
            if g.returncode != 0:
                raise RuntimeError(f"synth shard {idx} download failed:\n{g.stdout}\n{g.stderr}")

            for src in shard_out.glob("iter_*_content.txt"):
                dst = synth_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
            for src in shard_out.glob("iter_*_context.json"):
                dst = synth_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_shards) as ex:
            futs = [ex.submit(run_one_shard, i, shard_iters) for i, shard_iters in enumerate(shards)]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    # Rebuild all_samples.json from current local synth contents.
    samples = []
    for pth in sorted(synth_dir.glob("iter_*_content.txt")):
        m = re.match(r"iter_(\d+)_content\.txt$", pth.name)
        if not m:
            continue
        tid = int(m.group(1))
        samples.append({"task_id": f"iter_{tid:03d}", "content": pth.read_text()})
    (synth_dir / "all_samples.json").write_text(json.dumps(samples, indent=2))

    # Publish merged synth dir so monitor can read aggregate progress remotely.
    run([MODAL, "volume", "put", "-f", "atlas-openevolve-outputs", str(synth_dir), f"/synth/{synth_name}"], env, check=True)

    if not (p1.exists() and p2.exists() and best.exists()):
        run(
            [
                PYTHON,
                "-m",
                "experiment.build_fft_speedup_sft_datasets",
                "--trace",
                str(trace),
                "--synth-dir",
                str(synth_dir),
                "--problem-id",
                problem,
                "--out-p1",
                str(p1),
                "--out-p2",
                str(p2),
                "--out-best",
                str(best),
                "--correct-threshold",
                "0.99",
            ],
            env,
            check=True,
        )

    if method == "sft-best-traj":
        new_adapter = f"{run_prefix}_bestspeed_after_{problem}"
        cmd = [
            MODAL,
            "run",
            "experiment/train_atlas_sft.py",
            "--dataset",
            str(best),
            "--phase",
            "phase1",
            "--run-name",
            new_adapter,
            "--epochs",
            "3",
            "--learning-rate",
            "5e-5",
        ]
        if adapter_before:
            cmd += ["--resume-from", adapter_before]
        run(cmd, env, check=True)
        return new_adapter, None

    if method == "dpo-weighted-sft":
        p1_adapter = f"{run_prefix}_twostage_p1_after_{problem}"
        p2_adapter = f"{run_prefix}_twostage_p2_after_{problem}"
        cmd1 = [
            MODAL,
            "run",
            "experiment/train_atlas_sft.py",
            "--dataset",
            str(p1),
            "--phase",
            "phase1",
            "--run-name",
            p1_adapter,
            "--epochs",
            "3",
            "--learning-rate",
            "5e-5",
        ]
        if adapter_before:
            cmd1 += ["--resume-from", adapter_before]
        run(cmd1, env, check=True)
        run(
            [
                MODAL,
                "run",
                "experiment/train_atlas_sft.py",
                "--dataset",
                str(p2),
                "--phase",
                "phase2",
                "--resume-from",
                p1_adapter,
                "--run-name",
                p2_adapter,
                "--epochs",
                "3",
                "--learning-rate",
                "5e-5",
            ],
            env,
            check=True,
        )
        return p2_adapter, None

    # rlm
    scratch = local_dir / f"scratchpad_after_{problem}.txt"
    run([PYTHON, "experiment/build_rlm_memory_bank.py", "--in", str(p1), "--out", str(scratch)], env, check=True)
    remote = f"/streaming/{run_prefix}/rlm/{scratch.name}"
    run([MODAL, "volume", "put", "-f", "atlas-openevolve-outputs", str(scratch), remote], env, check=True)
    return None, str(scratch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="stream_perm_v1")
    ap.add_argument("--permutation-file", default="setup/stream_permutation_v1.txt")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll-secs", type=int, default=45)
    ap.add_argument("--modal-profile", default=os.environ.get("MODAL_PROFILE", "rishipython"))
    args = ap.parse_args()

    env = os.environ.copy()
    env["MODAL_PROFILE"] = args.modal_profile

    problems = [x.strip() for x in Path(args.permutation_file).read_text().splitlines() if x.strip()]
    eval_probs = problems[1:]
    methods = ("sft-best-traj", "dpo-weighted-sft", "rlm")

    state_path = REPO / "runs" / f"{args.run_prefix}_auto_chain" / "state.json"
    heartbeat_path = REPO / "runs" / f"{args.run_prefix}_auto_chain" / "heartbeat.json"
    errors_path = REPO / "runs" / f"{args.run_prefix}_auto_chain" / "errors.log"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "methods": {
                "sft-best-traj": {"adapter": "atlas_fftconv_bestspeed_only_v4", "rlm_bank": None, "completed": []},
                "dpo-weighted-sft": {"adapter": "atlas_fftconv_dpo_p2bridge_v4", "rlm_bank": None, "completed": []},
                "rlm": {"adapter": None, "rlm_bank": "data/rlm_memory_fftconv_synth_v1.txt", "completed": []},
            }
        }
        state_path.write_text(json.dumps(state, indent=2))

    lock = threading.Lock()

    def save_state() -> None:
        with lock:
            state_path.write_text(json.dumps(state, indent=2))

    def beat(method: str, stage: str, extra: dict | None = None) -> None:
        payload = {
            "ts": time.time(),
            "method": method,
            "stage": stage,
        }
        if extra:
            payload.update(extra)
        with lock:
            heartbeat_path.write_text(json.dumps(payload, indent=2))

    def log_error(method: str, exc: Exception) -> None:
        msg = (
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] method={method}\n"
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
        )
        with lock:
            with errors_path.open("a") as f:
                f.write(msg)

    def worker(method: str) -> None:
        while True:
            try:
                with lock:
                    mstate = state["methods"][method]
                    completed = set(mstate["completed"])
                    adapter = mstate.get("adapter")
                    rlm_bank = mstate.get("rlm_bank")

                # Find first incomplete problem for this method.
                idx = None
                for i, p in enumerate(eval_probs):
                    if p not in completed:
                        idx = i
                        break
                if idx is None:
                    beat(method, "completed_all")
                    return

                prob = eval_probs[idx]
                run_name = f"{args.run_prefix}_{method}_{prob}_oe"
                beat(method, "ensure_oe", {"problem": prob, "run_name": run_name})
                ensure_oe_running(env, run_name, prob, args.seed, args.iterations, adapter, rlm_bank)

                # Poll this OE leg to completion.
                while True:
                    n = get_trace_count(env, run_name)
                    beat(method, "poll_oe", {"problem": prob, "run_name": run_name, "trace_count": n})
                    if n >= args.iterations:
                        break
                    time.sleep(args.poll_secs)

                next_prob = eval_probs[idx + 1] if idx + 1 < len(eval_probs) else None
                beat(method, "build_update", {"problem": prob, "next_problem": next_prob})
                new_adapter, new_rlm = build_method_update(
                    env,
                    args.run_prefix,
                    method,
                    prob,
                    next_prob,
                    adapter,
                    rlm_bank,
                )

                with lock:
                    mstate = state["methods"][method]
                    if new_adapter:
                        mstate["adapter"] = new_adapter
                    if new_rlm:
                        mstate["rlm_bank"] = new_rlm
                    done = set(mstate["completed"])
                    done.add(prob)
                    mstate["completed"] = sorted(done)
                    state_path.write_text(json.dumps(state, indent=2))

                if next_prob:
                    with lock:
                        mstate = state["methods"][method]
                        adapter2 = mstate.get("adapter")
                        rlm2 = mstate.get("rlm_bank")
                    next_run = f"{args.run_prefix}_{method}_{next_prob}_oe"
                    beat(method, "launch_next_oe", {"problem": next_prob, "run_name": next_run})
                    ensure_oe_running(env, next_run, next_prob, args.seed, args.iterations, adapter2, rlm2)
            except Exception as exc:
                log_error(method, exc)
                beat(method, "error_retrying", {"error": str(exc)})
                time.sleep(30)

    threads = [threading.Thread(target=worker, args=(m,), daemon=False) for m in methods]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

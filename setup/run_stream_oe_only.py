#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PY = "/Users/rishi/miniconda3/envs/atlas/bin/python"
MODAL = "/Users/rishi/miniconda3/envs/atlas/bin/modal"


def run(cmd: list[str], *, check: bool = True, capture: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=e,
    )
    if check and p.returncode != 0:
        out = p.stdout or ""
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return p


def notify(msg: str, enabled: bool = True) -> None:
    print(msg, flush=True)
    if not enabled:
        return
    script = REPO / "send_update_over_text.py"
    if script.exists():
        try:
            run([PY, str(script), msg], check=False, capture=True)
        except Exception:
            pass


def modal_env(profile: str) -> dict[str, str]:
    return {"MODAL_PROFILE": profile}


def launch_detached_modal(args: list[str], profile: str) -> str:
    before = set(app_states(profile).keys())
    run([MODAL, "run", "-d", *args], env=modal_env(profile), capture=False)
    # Diff app ids to identify the newly created detached app.
    for _ in range(30):
        now = app_states(profile)
        new_ids = [x for x in now.keys() if x not in before]
        if new_ids:
            # Pick newest by creation timestamp when multiple apps appear.
            new_ids.sort(key=lambda x: now[x].get("Created at", ""))
            return new_ids[-1]
        time.sleep(1)
    raise RuntimeError("Could not identify newly launched app id after detached run.")


def app_states(profile: str) -> dict[str, dict[str, Any]]:
    p = run([MODAL, "app", "list", "--json"], env=modal_env(profile))
    arr = json.loads(p.stdout or "[]")
    return {x["App ID"]: x for x in arr}


def wait_for_apps(app_ids: list[str], profile: str, poll_s: int = 30) -> None:
    pending = set(app_ids)
    while pending:
        apps = app_states(profile)
        done: list[str] = []
        for app_id in sorted(pending):
            rec = apps.get(app_id)
            if rec is None:
                done.append(app_id)
                continue
            state = str(rec.get("State", ""))
            if state.startswith("stopped"):
                done.append(app_id)
        for d in done:
            pending.remove(d)
        if pending:
            print(f"[wait] apps pending={len(pending)} ids={sorted(pending)}", flush=True)
            time.sleep(poll_s)


def volume_exists(profile: str, volume: str, remote_path: str) -> bool:
    p = run([MODAL, "volume", "ls", volume, remote_path], env=modal_env(profile), check=False)
    return p.returncode == 0


def volume_get(profile: str, volume: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run([MODAL, "volume", "get", volume, remote_path, str(local_path)], env=modal_env(profile))


def volume_put(profile: str, volume: str, local_path: Path, remote_path: str) -> None:
    run([MODAL, "volume", "put", "-f", volume, str(local_path), remote_path], env=modal_env(profile))


def run_modal_sync(args: list[str], profile: str) -> None:
    p = run([MODAL, "run", *args], env=modal_env(profile), check=False)
    out = p.stdout or ""
    print(out, flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"modal run failed: {' '.join(args)}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_simple_rlm_bank(sft_p1_jsonl: Path, out_txt: Path, problem_id: str) -> None:
    rows = []
    for ln in sft_p1_jsonl.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    def key_fn(r: dict[str, Any]) -> tuple[float, float]:
        m = (r.get("_meta") or {})
        return (float(m.get("correctness", 0.0) or 0.0), float(m.get("speedup", 0.0) or 0.0))

    rows.sort(key=key_fn, reverse=True)
    top = rows[:20]
    parts = [
        f"# RLM scratchpad for {problem_id}\n",
        f"# Entries: {len(top)} (sorted by correctness, speedup)\n\n",
    ]
    for i, r in enumerate(top, start=1):
        m = (r.get("_meta") or {})
        msgs = r.get("messages") or []
        thinking = ""
        code = ""
        if len(msgs) >= 3:
            a = msgs[2] or {}
            thinking = str(a.get("thinking", "") or "")
            code = str(a.get("content", "") or "")
        thinking = " ".join(thinking.split())[:1200]
        code_lines = "\n".join(code.splitlines()[:40])
        parts.append(
            "=" * 40
            + "\n"
            + f"ENTRY {i} [correctness={float(m.get('correctness', 0.0) or 0.0):.3f}, "
            + f"speedup={float(m.get('speedup', 0.0) or 0.0):.4f}, iter={m.get('iteration', -1)}]\n"
            + "=" * 40
            + "\n"
            + "LESSON\n"
            + (thinking or "(no thinking captured)")
            + "\n\nCODE\n```python\n"
            + code_lines
            + "\n```\n\n"
        )
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("".join(parts))


def launch_oe(profile: str, run_name: str, problem_id: str, iterations: int, seed: int, adapter: str | None, rlm_bank: Path | None) -> str:
    cmd = [
        "experiment/openevolve_runner.py::run_evolution",
        "--problem-id", problem_id,
        "--task-family", "algotune",
        "--iterations", str(iterations),
        "--random-seed", str(seed),
        "--run-name", run_name,
    ]
    if adapter:
        cmd += ["--adapter-name", adapter]
    if rlm_bank:
        memory = rlm_bank.read_text()
        cmd += ["--rlm-memory-bank-text", memory]
    return launch_detached_modal(cmd, profile)


def run_sync_oe(profile: str, run_name: str, problem_id: str, iterations: int, seed: int, adapter: str | None, rlm_bank: Path | None) -> None:
    cmd = [
        MODAL,
        "run",
        "experiment/openevolve_runner.py",
        "--task-family", "algotune",
        "--problem-id", problem_id,
        "--iterations", str(iterations),
        "--random-seed", str(seed),
        "--run-name", run_name,
    ]
    if adapter:
        cmd += ["--adapter-name", adapter]
    if rlm_bank:
        cmd += ["--rlm-memory-bank", str(rlm_bank)]
    run(cmd, env=modal_env(profile), check=True, capture=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="stream_oe5_v1")
    ap.add_argument("--problems", default="fft_convolution,affine_transform_2d,convolve_1d,matrix_multiplication,outer_product")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modal-profile", default=os.environ.get("MODAL_PROFILE", "rishipython"))
    ap.add_argument("--poll-secs", type=int, default=30)
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    notify_enabled = not args.no_notify
    problems = [x.strip() for x in args.problems.split(",") if x.strip()]
    assert len(problems) == 5, "This runner is configured for exactly 5 problems."

    run_dir = REPO / "runs" / args.run_prefix
    state_path = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    state = load_json(state_path) or {
        "run_prefix": args.run_prefix,
        "problems": problems,
        "iterations": args.iterations,
        "seed": args.seed,
        "profile": args.modal_profile,
        "methods": {
            "base": {"type": "base", "adapter": None, "rlm_bank": None, "steps": []},
            "sft-best-traj": {"type": "bestspeed", "adapter": None, "rlm_bank": None, "steps": []},
            "dpo-weighted-sft": {"type": "twostage", "adapter": None, "rlm_bank": None, "steps": []},
            "rlm": {"type": "rlm", "adapter": None, "rlm_bank": None, "steps": []},
        },
        "watchdog": {"adapters_verified": [], "scratchpads_verified": []},
    }
    save_json(state_path, state)

    # Stage A: launch base for all 5 + problem1 for non-base methods in parallel.
    launch_batch: list[tuple[str, str, str, str]] = []
    for pid in problems:
        run_name = f"{args.run_prefix}_base_{pid}_oe"
        if volume_exists(args.modal_profile, "atlas-openevolve-outputs", f"/{run_name}/summary.json"):
            continue
        app_id = launch_oe(args.modal_profile, run_name, pid, args.iterations, args.seed, adapter=None, rlm_bank=None)
        launch_batch.append((app_id, "base", pid, run_name))

    p1 = problems[0]
    for method in ("sft-best-traj", "dpo-weighted-sft", "rlm"):
        run_name = f"{args.run_prefix}_{method.replace('-', '_')}_{p1}_oe"
        if volume_exists(args.modal_profile, "atlas-openevolve-outputs", f"/{run_name}/summary.json"):
            continue
        method_state = state["methods"][method]
        rlm_bank = Path(method_state["rlm_bank"]) if method_state.get("rlm_bank") else None
        app_id = launch_oe(
            args.modal_profile,
            run_name,
            p1,
            args.iterations,
            args.seed,
            adapter=method_state.get("adapter"),
            rlm_bank=rlm_bank,
        )
        launch_batch.append((app_id, method, p1, run_name))

    if launch_batch:
        notify(f"[stream:{args.run_prefix}] launched {len(launch_batch)} initial OE apps", notify_enabled)
        wait_for_apps([x[0] for x in launch_batch], args.modal_profile, poll_s=args.poll_secs)

    # Stage B: per-method streaming updates.
    for method in ("sft-best-traj", "dpo-weighted-sft", "rlm"):
        method_state = state["methods"][method]
        for i, pid in enumerate(problems):
            run_name = f"{args.run_prefix}_{method.replace('-', '_')}_{pid}_oe"

            # Ensure OE run exists/completed.
            if not volume_exists(args.modal_profile, "atlas-openevolve-outputs", f"/{run_name}/summary.json"):
                rlm_bank = Path(method_state["rlm_bank"]) if method_state.get("rlm_bank") else None
                app_id = launch_oe(
                    args.modal_profile,
                    run_name,
                    pid,
                    args.iterations,
                    args.seed,
                    adapter=method_state.get("adapter"),
                    rlm_bank=rlm_bank,
                )
                notify(f"[stream:{args.run_prefix}] launch OE {method} on {pid} app={app_id}", notify_enabled)
                wait_for_apps([app_id], args.modal_profile, poll_s=args.poll_secs)

            if i == len(problems) - 1:
                method_state["steps"].append({"problem": pid, "oe_run": run_name, "updated_for_next": False})
                save_json(state_path, state)
                continue

            # Distill/update for next problem.
            local_dir = run_dir / method / pid
            trace_local = local_dir / "evolution_trace.jsonl"
            synth_local = local_dir / "synth"
            p1_jsonl = local_dir / "p1.jsonl"
            p2_jsonl = local_dir / "p2.jsonl"
            best_jsonl = local_dir / "best.jsonl"

            if not trace_local.exists():
                volume_get(
                    args.modal_profile,
                    "atlas-openevolve-outputs",
                    f"/{run_name}/oe/evolution_trace.jsonl",
                    trace_local,
                )

            synth_name = f"{args.run_prefix}_{method.replace('-', '_')}_{pid}_synth"
            if not (synth_local / "all_samples.json").exists():
                run_modal_sync(
                    [
                        "experiment/synth_reasoning.py::main_algotune",
                        "--trace-path", str(trace_local),
                        "--problem-id", pid,
                        "--out-name", synth_name,
                    ],
                    args.modal_profile,
                )
                volume_get(
                    args.modal_profile,
                    "atlas-openevolve-outputs",
                    f"/synth/{synth_name}",
                    synth_local,
                )

            if not (p1_jsonl.exists() and p2_jsonl.exists() and best_jsonl.exists()):
                run(
                    [
                        PY,
                        "-m", "experiment.build_fft_speedup_sft_datasets",
                        "--trace", str(trace_local),
                        "--synth-dir", str(synth_local),
                        "--problem-id", pid,
                        "--out-p1", str(p1_jsonl),
                        "--out-p2", str(p2_jsonl),
                        "--out-best", str(best_jsonl),
                        "--correct-threshold", "0.99",
                    ],
                    check=True,
                    capture=True,
                )

            if method == "sft-best-traj":
                adapter_name = f"{args.run_prefix}_bestspeed_after_{pid}"
                train_cmd = [
                    "experiment/train_atlas_sft.py",
                    "--dataset", str(best_jsonl),
                    "--phase", "phase1",
                    "--run-name", adapter_name,
                    "--epochs", "3",
                    "--learning-rate", "5e-5",
                ]
                if method_state.get("adapter"):
                    train_cmd += ["--resume-from", method_state["adapter"]]
                run_modal_sync(train_cmd, args.modal_profile)
                if not volume_exists(args.modal_profile, "atlas-models", f"/{adapter_name}/training_summary.json"):
                    raise RuntimeError(f"watchdog: missing adapter summary for {adapter_name}")
                state["watchdog"]["adapters_verified"].append(adapter_name)
                method_state["adapter"] = adapter_name

            elif method == "dpo-weighted-sft":
                p1_adapter = f"{args.run_prefix}_twostage_p1_after_{pid}"
                p2_adapter = f"{args.run_prefix}_twostage_p2_after_{pid}"

                train_p1 = [
                    "experiment/train_atlas_sft.py",
                    "--dataset", str(p1_jsonl),
                    "--phase", "phase1",
                    "--run-name", p1_adapter,
                    "--epochs", "3",
                    "--learning-rate", "5e-5",
                ]
                if method_state.get("adapter"):
                    train_p1 += ["--resume-from", method_state["adapter"]]
                run_modal_sync(train_p1, args.modal_profile)

                train_p2 = [
                    "experiment/train_atlas_sft.py",
                    "--dataset", str(p2_jsonl),
                    "--phase", "phase2",
                    "--resume-from", p1_adapter,
                    "--run-name", p2_adapter,
                    "--epochs", "3",
                    "--learning-rate", "5e-5",
                ]
                run_modal_sync(train_p2, args.modal_profile)

                if not volume_exists(args.modal_profile, "atlas-models", f"/{p2_adapter}/training_summary.json"):
                    raise RuntimeError(f"watchdog: missing adapter summary for {p2_adapter}")
                state["watchdog"]["adapters_verified"].append(p2_adapter)
                method_state["adapter"] = p2_adapter

            else:  # rlm
                rlm_local = local_dir / f"scratchpad_after_{pid}.txt"
                build_simple_rlm_bank(p1_jsonl, rlm_local, pid)
                rlm_remote = f"/streaming/{args.run_prefix}/rlm/{rlm_local.name}"
                volume_put(args.modal_profile, "atlas-openevolve-outputs", rlm_local, rlm_remote)
                if not volume_exists(args.modal_profile, "atlas-openevolve-outputs", rlm_remote):
                    raise RuntimeError(f"watchdog: missing scratchpad upload {rlm_remote}")
                state["watchdog"]["scratchpads_verified"].append(rlm_remote)
                method_state["rlm_bank"] = str(rlm_local)

            method_state["steps"].append({
                "problem": pid,
                "oe_run": run_name,
                "updated_for_next": True,
                "adapter": method_state.get("adapter"),
                "rlm_bank": method_state.get("rlm_bank"),
            })
            save_json(state_path, state)
            notify(f"[stream:{args.run_prefix}] updated {method} after {pid}", notify_enabled)

    save_json(state_path, state)
    notify(f"[stream:{args.run_prefix}] complete for problems={','.join(problems)}", notify_enabled)


if __name__ == "__main__":
    main()

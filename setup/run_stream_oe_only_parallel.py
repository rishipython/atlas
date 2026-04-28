#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/Users/rishi/miniconda3/envs/atlas/bin/python"
MODAL = "/Users/rishi/miniconda3/envs/atlas/bin/modal"


class Orchestrator:
    def __init__(self, run_prefix: str, problems: list[str], iterations: int, seed: int, profile: str):
        self.run_prefix = run_prefix
        self.problems = problems
        self.iterations = iterations
        self.seed = seed
        self.profile = profile
        self.run_dir = REPO / "runs" / run_prefix
        self.log_dir = self.run_dir / "logs"
        self.state_path = self.run_dir / "state.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_lock = threading.Lock()
        self.sem = threading.Semaphore(10)  # hard cap for concurrent GPU jobs
        self.state: dict[str, Any] = self._load_state()

    def _env(self) -> dict[str, str]:
        e = os.environ.copy()
        e["MODAL_PROFILE"] = self.profile
        return e

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        st = {
            "run_prefix": self.run_prefix,
            "problems": self.problems,
            "iterations": self.iterations,
            "seed": self.seed,
            "profile": self.profile,
            "methods": {
                "base": {"adapter": None, "rlm_bank": None, "steps": []},
                "sft-best-traj": {"adapter": None, "rlm_bank": None, "steps": []},
                "dpo-weighted-sft": {"adapter": None, "rlm_bank": None, "steps": []},
                "rlm": {"adapter": None, "rlm_bank": None, "steps": []},
            },
            "watchdog": {"adapters_verified": [], "scratchpads_verified": []},
        }
        self.state_path.write_text(json.dumps(st, indent=2))
        return st

    def _save_state(self) -> None:
        with self.state_lock:
            self.state_path.write_text(json.dumps(self.state, indent=2))

    def _run(self, cmd: list[str], log_file: Path | None = None, gpu_job: bool = False) -> None:
        if gpu_job:
            self.sem.acquire()
        try:
            if log_file is None:
                p = subprocess.run(cmd, cwd=str(REPO), env=self._env(), text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
                return
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a") as f:
                f.write(f"\n$ {' '.join(cmd)}\n")
                f.flush()
                p = subprocess.run(
                    cmd,
                    cwd=str(REPO),
                    env=self._env(),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                f.flush()
            if p.returncode != 0:
                raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)} (see {log_file})")
        finally:
            if gpu_job:
                self.sem.release()

    def _vol_exists(self, volume: str, remote_path: str) -> bool:
        p = subprocess.run(
            [MODAL, "volume", "ls", volume, remote_path],
            cwd=str(REPO),
            env=self._env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return p.returncode == 0

    def _vol_get(self, volume: str, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._run([MODAL, "volume", "get", volume, remote_path, str(local_path)])

    def _vol_put(self, volume: str, local_path: Path, remote_path: str) -> None:
        self._run([MODAL, "volume", "put", "-f", volume, str(local_path), remote_path])

    def _run_oe(self, method: str, problem: str, adapter: str | None, rlm_bank: Path | None) -> str:
        run_name = f"{self.run_prefix}_{method.replace('-', '_')}_{problem}_oe"
        done_flag = f"/{run_name}/summary.json"
        if self._vol_exists("atlas-openevolve-outputs", done_flag):
            return run_name
        logf = self.log_dir / f"oe_{method}_{problem}.log"
        cmd = [
            MODAL,
            "run",
            "experiment/openevolve_runner.py",
            "--task-family",
            "algotune",
            "--problem-id",
            problem,
            "--iterations",
            str(self.iterations),
            "--random-seed",
            str(self.seed),
            "--run-name",
            run_name,
        ]
        if adapter:
            cmd += ["--adapter-name", adapter]
        if rlm_bank and rlm_bank.exists():
            cmd += ["--rlm-memory-bank", str(rlm_bank)]
        self._run(cmd, log_file=logf, gpu_job=True)
        return run_name

    def _build_rlm_bank(self, p1_jsonl: Path, out_txt: Path, problem_id: str) -> None:
        rows = []
        for ln in p1_jsonl.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
        rows.sort(
            key=lambda r: (
                float((r.get("_meta") or {}).get("correctness", 0.0) or 0.0),
                float((r.get("_meta") or {}).get("speedup", 0.0) or 0.0),
            ),
            reverse=True,
        )
        parts = [f"# RLM scratchpad for {problem_id}\n"]
        for i, r in enumerate(rows[:20], start=1):
            m = r.get("_meta") or {}
            a = (r.get("messages") or [{}, {}, {}])[2]
            thinking = " ".join(str(a.get("thinking", "") or "").split())[:1200]
            code = "\n".join(str(a.get("content", "") or "").splitlines()[:40])
            parts.append(
                f"\nENTRY {i} [correctness={float(m.get('correctness', 0.0)):.3f}, speedup={float(m.get('speedup', 0.0)):.4f}, iter={m.get('iteration', -1)}]\n"
                f"LESSON\n{thinking}\n\nCODE\n```python\n{code}\n```\n"
            )
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(parts))

    def _update_method_after_problem(self, method: str, problem: str, run_name: str) -> None:
        i = self.problems.index(problem)
        if i >= len(self.problems) - 1:
            return
        local_dir = self.run_dir / method / problem
        trace_local = local_dir / "evolution_trace.jsonl"
        synth_local = local_dir / "synth"
        p1_jsonl = local_dir / "p1.jsonl"
        p2_jsonl = local_dir / "p2.jsonl"
        best_jsonl = local_dir / "best.jsonl"

        if not trace_local.exists():
            self._vol_get("atlas-openevolve-outputs", f"/{run_name}/oe/evolution_trace.jsonl", trace_local)

        synth_name = f"{self.run_prefix}_{method.replace('-', '_')}_{problem}_synth"
        if not (synth_local / "all_samples.json").exists():
            self._run(
                [
                    MODAL,
                    "run",
                    "experiment/synth_reasoning.py::main_algotune",
                    "--trace-path",
                    str(trace_local),
                    "--problem-id",
                    problem,
                    "--out-name",
                    synth_name,
                ],
                log_file=self.log_dir / f"synth_{method}_{problem}.log",
                gpu_job=True,
            )
            self._vol_get("atlas-openevolve-outputs", f"/synth/{synth_name}", synth_local)

        if not (p1_jsonl.exists() and p2_jsonl.exists() and best_jsonl.exists()):
            self._run(
                [
                    PYTHON,
                    "-m",
                    "experiment.build_fft_speedup_sft_datasets",
                    "--trace",
                    str(trace_local),
                    "--synth-dir",
                    str(synth_local),
                    "--problem-id",
                    problem,
                    "--out-p1",
                    str(p1_jsonl),
                    "--out-p2",
                    str(p2_jsonl),
                    "--out-best",
                    str(best_jsonl),
                    "--correct-threshold",
                    "0.99",
                ],
                log_file=self.log_dir / f"build_{method}_{problem}.log",
            )

        if method == "sft-best-traj":
            prev = self.state["methods"][method].get("adapter")
            adapter_name = f"{self.run_prefix}_bestspeed_after_{problem}"
            cmd = [
                MODAL,
                "run",
                "experiment/train_atlas_sft.py",
                "--dataset",
                str(best_jsonl),
                "--phase",
                "phase1",
                "--run-name",
                adapter_name,
                "--epochs",
                "3",
                "--learning-rate",
                "5e-5",
            ]
            if prev:
                cmd += ["--resume-from", prev]
            self._run(cmd, log_file=self.log_dir / f"train_{method}_{problem}.log", gpu_job=True)
            if not self._vol_exists("atlas-models", f"/{adapter_name}/training_summary.json"):
                raise RuntimeError(f"watchdog failed adapter missing: {adapter_name}")
            self.state["methods"][method]["adapter"] = adapter_name
            self.state["watchdog"]["adapters_verified"].append(adapter_name)

        elif method == "dpo-weighted-sft":
            prev = self.state["methods"][method].get("adapter")
            p1_adapter = f"{self.run_prefix}_twostage_p1_after_{problem}"
            p2_adapter = f"{self.run_prefix}_twostage_p2_after_{problem}"

            c1 = [
                MODAL,
                "run",
                "experiment/train_atlas_sft.py",
                "--dataset",
                str(p1_jsonl),
                "--phase",
                "phase1",
                "--run-name",
                p1_adapter,
                "--epochs",
                "3",
                "--learning-rate",
                "5e-5",
            ]
            if prev:
                c1 += ["--resume-from", prev]
            self._run(c1, log_file=self.log_dir / f"train_{method}_{problem}_p1.log", gpu_job=True)
            c2 = [
                MODAL,
                "run",
                "experiment/train_atlas_sft.py",
                "--dataset",
                str(p2_jsonl),
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
            ]
            self._run(c2, log_file=self.log_dir / f"train_{method}_{problem}_p2.log", gpu_job=True)
            if not self._vol_exists("atlas-models", f"/{p2_adapter}/training_summary.json"):
                raise RuntimeError(f"watchdog failed adapter missing: {p2_adapter}")
            self.state["methods"][method]["adapter"] = p2_adapter
            self.state["watchdog"]["adapters_verified"].append(p2_adapter)

        elif method == "rlm":
            rlm_local = local_dir / f"scratchpad_after_{problem}.txt"
            self._build_rlm_bank(p1_jsonl, rlm_local, problem)
            remote = f"/streaming/{self.run_prefix}/rlm/{rlm_local.name}"
            self._vol_put("atlas-openevolve-outputs", rlm_local, remote)
            if not self._vol_exists("atlas-openevolve-outputs", remote):
                raise RuntimeError(f"watchdog failed scratchpad missing: {remote}")
            self.state["methods"][method]["rlm_bank"] = str(rlm_local)
            self.state["watchdog"]["scratchpads_verified"].append(remote)

        self._save_state()

    def _run_base_parallel(self) -> None:
        def job(pid: str) -> str:
            return self._run_oe("base", pid, adapter=None, rlm_bank=None)

        with ThreadPoolExecutor(max_workers=min(5, len(self.problems))) as ex:
            futs = [ex.submit(job, p) for p in self.problems]
            for fut in as_completed(futs):
                rn = fut.result()
                prob = rn.split("_base_", 1)[1].rsplit("_oe", 1)[0]
                with self.state_lock:
                    self.state["methods"]["base"]["steps"].append({"problem": prob, "oe_run": rn, "done": True})
                    self._save_state()

    def _run_method_chain(self, method: str) -> None:
        # Method chains are evaluated on the last N-1 problems.
        for pid in self.problems[1:]:
            adapter = self.state["methods"][method].get("adapter")
            rlm_bank_s = self.state["methods"][method].get("rlm_bank")
            rlm_bank = Path(rlm_bank_s) if rlm_bank_s else None
            run_name = self._run_oe(method, pid, adapter=adapter, rlm_bank=rlm_bank)
            with self.state_lock:
                self.state["methods"][method]["steps"].append(
                    {
                        "problem": pid,
                        "oe_run": run_name,
                        "adapter_before": adapter,
                        "rlm_before": rlm_bank_s,
                    }
                )
                self._save_state()
            self._update_method_after_problem(method, pid, run_name)

    def _bootstrap_from_base_fft(self) -> None:
        """Build initial method states from a single shared base FFT OE run."""
        fft = self.problems[0]
        run_name = self._run_oe("base", fft, adapter=None, rlm_bank=None)
        local_dir = self.run_dir / "bootstrap_base_fft"
        trace_local = local_dir / "evolution_trace.jsonl"
        synth_local = local_dir / "synth"
        p1_jsonl = local_dir / "p1.jsonl"
        p2_jsonl = local_dir / "p2.jsonl"
        best_jsonl = local_dir / "best.jsonl"

        if not trace_local.exists():
            self._vol_get("atlas-openevolve-outputs", f"/{run_name}/oe/evolution_trace.jsonl", trace_local)

        synth_name = f"{self.run_prefix}_bootstrap_base_{fft}_synth"
        if not (synth_local / "all_samples.json").exists():
            self._run(
                [
                    MODAL,
                    "run",
                    "experiment/synth_reasoning.py::main_algotune",
                    "--trace-path",
                    str(trace_local),
                    "--problem-id",
                    fft,
                    "--out-name",
                    synth_name,
                ],
                log_file=self.log_dir / "bootstrap_synth.log",
                gpu_job=True,
            )
            self._vol_get("atlas-openevolve-outputs", f"/synth/{synth_name}", synth_local)

        if not (p1_jsonl.exists() and p2_jsonl.exists() and best_jsonl.exists()):
            self._run(
                [
                    PYTHON,
                    "-m",
                    "experiment.build_fft_speedup_sft_datasets",
                    "--trace",
                    str(trace_local),
                    "--synth-dir",
                    str(synth_local),
                    "--problem-id",
                    fft,
                    "--out-p1",
                    str(p1_jsonl),
                    "--out-p2",
                    str(p2_jsonl),
                    "--out-best",
                    str(best_jsonl),
                    "--correct-threshold",
                    "0.99",
                ],
                log_file=self.log_dir / "bootstrap_build.log",
            )

        # Build SFT best-traj bootstrap adapter
        sft_adapter = f"{self.run_prefix}_bootstrap_bestspeed_after_{fft}"
        if not self._vol_exists("atlas-models", f"/{sft_adapter}/training_summary.json"):
            self._run(
                [
                    MODAL,
                    "run",
                    "experiment/train_atlas_sft.py",
                    "--dataset",
                    str(best_jsonl),
                    "--phase",
                    "phase1",
                    "--run-name",
                    sft_adapter,
                    "--epochs",
                    "3",
                    "--learning-rate",
                    "5e-5",
                ],
                log_file=self.log_dir / "bootstrap_train_sft.log",
                gpu_job=True,
            )
        self.state["methods"]["sft-best-traj"]["adapter"] = sft_adapter

        # Build DPO-weighted two-stage bootstrap adapter
        p1_adapter = f"{self.run_prefix}_bootstrap_twostage_p1_after_{fft}"
        p2_adapter = f"{self.run_prefix}_bootstrap_twostage_p2_after_{fft}"
        if not self._vol_exists("atlas-models", f"/{p2_adapter}/training_summary.json"):
            if not self._vol_exists("atlas-models", f"/{p1_adapter}/training_summary.json"):
                self._run(
                    [
                        MODAL,
                        "run",
                        "experiment/train_atlas_sft.py",
                        "--dataset",
                        str(p1_jsonl),
                        "--phase",
                        "phase1",
                        "--run-name",
                        p1_adapter,
                        "--epochs",
                        "3",
                        "--learning-rate",
                        "5e-5",
                    ],
                    log_file=self.log_dir / "bootstrap_train_dpo_p1.log",
                    gpu_job=True,
                )
            self._run(
                [
                    MODAL,
                    "run",
                    "experiment/train_atlas_sft.py",
                    "--dataset",
                    str(p2_jsonl),
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
                log_file=self.log_dir / "bootstrap_train_dpo_p2.log",
                gpu_job=True,
            )
        self.state["methods"]["dpo-weighted-sft"]["adapter"] = p2_adapter

        # Build RLM bootstrap memory bank
        rlm_local = local_dir / f"scratchpad_after_{fft}.txt"
        self._build_rlm_bank(p1_jsonl, rlm_local, fft)
        remote = f"/streaming/{self.run_prefix}/rlm/{rlm_local.name}"
        self._vol_put("atlas-openevolve-outputs", rlm_local, remote)
        self.state["methods"]["rlm"]["rlm_bank"] = str(rlm_local)
        self._save_state()

    def run_all(self) -> None:
        # Shared bootstrap from base OE on first problem.
        self._bootstrap_from_base_fft()

        workers = []
        t_base = threading.Thread(target=self._run_base_parallel, name="base-parallel")
        workers.append(t_base)
        t_sft = threading.Thread(target=self._run_method_chain, args=("sft-best-traj",), name="sft-chain")
        t_dpo = threading.Thread(target=self._run_method_chain, args=("dpo-weighted-sft",), name="dpo-chain")
        t_rlm = threading.Thread(target=self._run_method_chain, args=("rlm",), name="rlm-chain")
        workers.extend([t_sft, t_dpo, t_rlm])

        for t in workers:
            t.start()
        for t in workers:
            t.join()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="stream_oe5_methods_v1")
    ap.add_argument("--problems", default="fft_convolution,affine_transform_2d,convolve_1d,matrix_multiplication,outer_product")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modal-profile", default=os.environ.get("MODAL_PROFILE", "rishipython"))
    args = ap.parse_args()

    probs = [x.strip() for x in args.problems.split(",") if x.strip()]
    if len(probs) != 5:
        raise ValueError("Expected exactly 5 problems for this runner.")

    orch = Orchestrator(args.run_prefix, probs, args.iterations, args.seed, args.modal_profile)
    orch.run_all()


if __name__ == "__main__":
    main()

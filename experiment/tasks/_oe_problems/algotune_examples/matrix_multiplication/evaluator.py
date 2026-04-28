"""
Evaluator for the matrix_multiplication task with baseline comparison.
Vendored-style evaluator: baseline task logic is loaded from local task_ref.py
in the same problem folder, not from an external AlgoTune package install.
"""

import concurrent.futures
import importlib.util
import time
import traceback
from pathlib import Path

import numpy as np
from openevolve.evaluation_result import EvaluationResult


def _with_timeout(fn, args=(), timeout_seconds=120):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args)
        return fut.result(timeout=timeout_seconds)


def _load_module(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(fn, problem, runs=3, warmup=1, timeout_seconds=120):
    for _ in range(warmup):
        try:
            _with_timeout(fn, args=(problem,), timeout_seconds=timeout_seconds)
        except Exception:
            pass
    times = []
    outs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = _with_timeout(fn, args=(problem,), timeout_seconds=timeout_seconds)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        outs.append(out)
    return float(np.min(times)), outs[0]


def evaluate(program_path, config=None):
    try:
        base_dir = Path(__file__).parent
        ref_mod = _load_module(base_dir / "task_ref.py", "task_ref")
        prog_mod = _load_module(Path(program_path), "program")

        if not hasattr(prog_mod, "run_solver"):
            return EvaluationResult(
                metrics={
                    "correctness": 0.0,
                    "correctness_score": 0.0,
                    "performance_score": 0.0,
                    "combined_score": 0.0,
                    "speedup": 0.0,
                    "speedup_score": 0.0,
                },
                artifacts={"feedback": "Missing run_solver function"},
            )

        task = ref_mod.MatrixMultiplication()
        num_trials = 5
        data_size = 100
        timeout_seconds = 120
        num_runs = 3
        warmup_runs = 1

        correctness_scores = []
        speedups = []
        baseline_times = []
        evolved_times = []

        for trial in range(num_trials):
            problem = task.generate_problem(n=data_size, random_seed=trial)
            base_ms, _ = _measure(task.solve, problem, runs=num_runs, warmup=warmup_runs, timeout_seconds=timeout_seconds)
            evo_ms, evo_out = _measure(prog_mod.run_solver, problem, runs=num_runs, warmup=warmup_runs, timeout_seconds=timeout_seconds)

            is_valid = False
            try:
                is_valid = bool(task.is_solution(problem, evo_out))
            except Exception:
                is_valid = False

            correctness_scores.append(1.0 if is_valid else 0.0)
            baseline_times.append(base_ms)
            evolved_times.append(evo_ms)
            if is_valid and evo_ms > 0:
                speedups.append(base_ms / evo_ms)

        avg_correctness = float(np.mean(correctness_scores)) if correctness_scores else 0.0
        mean_speedup = float(np.mean(speedups)) if speedups else 0.0
        combined = 0.7 * avg_correctness + 0.3 * min(mean_speedup, 5.0) / 5.0

        return EvaluationResult(
            metrics={
                "correctness": avg_correctness,
                "correctness_score": avg_correctness,
                "performance_score": float(np.mean([1.0 / (1.0 + t) for t in evolved_times])) if evolved_times else 0.0,
                "combined_score": float(combined),
                "speedup": mean_speedup,
                "speedup_score": mean_speedup,
            },
            artifacts={
                "baseline_comparison": {
                    "mean_speedup": mean_speedup,
                    "baseline_times": baseline_times,
                    "evolved_times": evolved_times,
                    "speedups": speedups,
                }
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "correctness": 0.0,
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "speedup": 0.0,
                "speedup_score": 0.0,
            },
            artifacts={"feedback": f"{e}\n{traceback.format_exc()}"},
        )

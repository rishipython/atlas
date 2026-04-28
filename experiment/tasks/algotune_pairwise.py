"""AlgoTune-style task family — speed up a reference Python function.

We give the LLM a deliberately-slow reference implementation of a common
numerical kernel and ask it to return a faster one with the same output.
Score = (reference_time / candidate_time) on a fixed validation set,
clipped to a cap so one-off outliers don't dominate.

Problems in this family:
  - ``pairwise_distance``: compute pairwise squared Euclidean distance
    matrix for ``X \\in R^{N,D}``. Naive Python triple-loop is ~100-1000x
    slower than the vectorised ``X @ X.T`` trick, so there's a big
    speedup ceiling and it's easy to be correct.

Unlike the kernel family, this task only needs CPU + numpy for eval (no
GPU, no Triton). Candidates that crash fail cleanly with a traceback.
Crash rate should be very low since all the evolved code is pure Python.
"""

from __future__ import annotations

from .base import TaskFamily, TaskSpec


# ---------------------------------------------------------------------------
# pairwise_distance
# ---------------------------------------------------------------------------
_PAIRWISE_INITIAL = '''\
"""AlgoTune: speed up pairwise squared Euclidean distances.

Given a 2-D numpy array X of shape (N, D), compute the full (N, N) matrix
D where D[i, j] = sum_k (X[i, k] - X[j, k])**2.

The reference below is a correct-but-very-slow triple-loop Python
implementation. Your task inside the EVOLVE-BLOCK is to replace it with
a much faster implementation that returns a numerically-equal result on
the evaluator's test matrices. You are free to use numpy, scipy, numba,
or any other package that is importable.
"""

# EVOLVE-BLOCK-START
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    N, D = X.shape
    out = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(D):
                diff = X[i, k] - X[j, k]
                s += diff * diff
            out[i, j] = s
    return out
# EVOLVE-BLOCK-END
'''


_PAIRWISE_EVALUATOR = '''\
"""OpenEvolve evaluator for AlgoTune pairwise_sq_dist."""
from __future__ import annotations

import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from openevolve.evaluation_result import EvaluationResult


# Validation + benchmark shapes. Kept small so the REFERENCE (triple loop)
# finishes within a few seconds; candidates are only expected to be faster.
_VAL_SHAPES = [(50, 8), (80, 16)]
_BENCH_SHAPES = [(200, 16), (400, 32), (300, 64)]
_RTOL = 1e-6
_ATOL = 1e-6
_MAX_REPEATS = 5


def _reference_impl(X):
    """Vectorised reference used only to build ground-truth values.

    Quadratic in N but in numpy, not Python — fast enough for our
    validation shapes (~1ms at N=80, D=16)."""
    sq = (X * X).sum(axis=1)
    G = X @ X.T
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    # Clamp tiny negatives from fp error.
    np.maximum(D2, 0.0, out=D2)
    return D2


def _reference_timed(X, timeout_s=5.0):
    """Time a pure-Python triple-loop (slow) on the given shape to produce
    the denominator for the speedup score. We cap at ``timeout_s`` and
    fall back to a per-row inner-numpy loop if the true triple-loop
    would take longer, so small shapes use the fair baseline and larger
    shapes use an at-least-as-slow fallback."""
    N, D = X.shape
    # For tiny shapes, do the honest triple-loop baseline.
    if N <= 60 and D <= 16:
        out = np.zeros((N, N), dtype=np.float64)
        t0 = time.perf_counter()
        for i in range(N):
            for j in range(N):
                s = 0.0
                for k in range(D):
                    diff = X[i, k] - X[j, k]
                    s += diff * diff
                out[i, j] = s
        return out, time.perf_counter() - t0
    # Larger shapes: use a nested per-row numpy baseline so the reference
    # stays slow-ish but bounded. Candidates still need to beat this.
    out = np.zeros((N, N), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N):
        diff = X - X[i]
        out[i] = (diff * diff).sum(axis=1)
    return out, time.perf_counter() - t0


def _run_candidate(program_path, X, timeout_s=30.0):
    out_path = program_path + ".oeout.pkl"
    np_path = program_path + ".oein.npz"
    np.savez(np_path, X=X)
    runner = (
        "import pickle, sys, traceback, time, importlib.util, numpy as np\\n"
        "data = np.load(" + repr(np_path) + ")\\n"
        "X = data['X']\\n"
        "spec = importlib.util.spec_from_file_location('prog', " + repr(program_path) + ")\\n"
        "mod = importlib.util.module_from_spec(spec)\\n"
        "try:\\n"
        "    spec.loader.exec_module(mod)\\n"
        "    # Warmup (JIT compiles etc.)\\n"
        "    _ = mod.pairwise_sq_dist(X)\\n"
        "    ts = []\\n"
        "    out = None\\n"
        "    for _ in range(5):\\n"
        "        t0 = time.perf_counter()\\n"
        "        out = mod.pairwise_sq_dist(X)\\n"
        "        ts.append(time.perf_counter() - t0)\\n"
        "    pickle.dump({'out': np.asarray(out), 'times': ts}, open(" + repr(out_path) + ", 'wb'))\\n"
        "except Exception as e:\\n"
        "    pickle.dump({'error': repr(e), 'tb': traceback.format_exc()}, open(" + repr(out_path) + ", 'wb'))\\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
        tf.write(runner)
        script = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if not os.path.exists(out_path):
            return {
                "error": f"no output; exit={proc.returncode}; stderr={proc.stderr[:2000]}"
            }
        with open(out_path, "rb") as f:
            return pickle.load(f)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s"}
    finally:
        for p in (script, out_path, np_path):
            if os.path.exists(p):
                os.unlink(p)


def evaluate(program_path):
    rng = np.random.default_rng(123)
    per_shape = []
    feedback_parts = []

    # Correctness pass on validation shapes
    for (N, D) in _VAL_SHAPES:
        X = rng.standard_normal((N, D))
        res = _run_candidate(program_path, X, timeout_s=30.0)
        if "error" in res:
            per_shape.append({"shape": (N, D), "correct": False, "error": res["error"]})
            feedback_parts.append(
                f"[val {N}x{D}] CRASH: {res['error'][:400]}"
            )
            continue
        out = np.asarray(res["out"], dtype=np.float64)
        ref = _reference_impl(X)
        if out.shape != ref.shape:
            per_shape.append({
                "shape": (N, D),
                "correct": False,
                "error": f"bad output shape {out.shape}, expected {ref.shape}",
            })
            feedback_parts.append(f"[val {N}x{D}] wrong shape: {out.shape}")
            continue
        if not np.allclose(out, ref, rtol=_RTOL, atol=_ATOL, equal_nan=False):
            max_err = float(np.max(np.abs(out - ref)))
            per_shape.append({
                "shape": (N, D),
                "correct": False,
                "error": f"numerical mismatch (max |err|={max_err:.3e})",
            })
            feedback_parts.append(
                f"[val {N}x{D}] wrong values (max |err|={max_err:.3e})"
            )
            continue
        per_shape.append({"shape": (N, D), "correct": True})
        feedback_parts.append(f"[val {N}x{D}] correct")

    if not all(r["correct"] for r in per_shape):
        # Don't bother benchmarking if any validation failed.
        return EvaluationResult(
            metrics={
                "correctness": 0.0,
                "speedup": 0.0,
                "combined_score": 0.15 if any("error" in r for r in per_shape) else 0.0,
            },
            artifacts={"feedback": "\\n".join(feedback_parts)[:6000]},
        )

    # Benchmark pass on larger shapes
    speedups = []
    for (N, D) in _BENCH_SHAPES:
        X = rng.standard_normal((N, D))
        _, ref_time = _reference_timed(X)
        res = _run_candidate(program_path, X, timeout_s=60.0)
        if "error" in res:
            per_shape.append({"shape": (N, D), "correct": False, "error": res["error"]})
            feedback_parts.append(
                f"[bench {N}x{D}] CRASH during timing: {res['error'][:400]}"
            )
            speedups.append(0.0)
            continue
        out = np.asarray(res["out"], dtype=np.float64)
        ref = _reference_impl(X)
        if out.shape != ref.shape or not np.allclose(
            out, ref, rtol=_RTOL, atol=_ATOL
        ):
            feedback_parts.append(f"[bench {N}x{D}] wrong values during timing")
            per_shape.append({"shape": (N, D), "correct": False, "error": "bench mismatch"})
            speedups.append(0.0)
            continue
        cand_time = float(min(res["times"]))
        speedup = ref_time / max(cand_time, 1e-9)
        speedups.append(speedup)
        per_shape.append({
            "shape": (N, D),
            "correct": True,
            "ref_time": ref_time,
            "cand_time": cand_time,
            "speedup": speedup,
        })
        feedback_parts.append(
            f"[bench {N}x{D}] {speedup:.1f}x (ref={ref_time*1e3:.2f}ms, cand={cand_time*1e3:.2f}ms)"
        )

    all_correct = all(r.get("correct") for r in per_shape)
    mean_speedup = float(np.mean(speedups)) if speedups else 0.0
    # log speedup to handle 1000x outliers gracefully; cap at log(100)
    capped = np.clip(speedups, 1e-3, 200.0)
    log_score = float(np.mean(np.log(capped))) if len(capped) else 0.0
    # combined_score in ~[0, 1]: 0 if broken, 0.5 at 10x, 1.0 at 100x+
    combined = 0.5 * (log_score / np.log(100.0))
    combined = float(np.clip(0.5 + combined, 0.05, 1.0) if all_correct else 0.15)

    return EvaluationResult(
        metrics={
            "correctness": 1.0 if all_correct else 0.0,
            "speedup": mean_speedup,
            "combined_score": combined,
        },
        artifacts={
            "feedback": "\\n".join(feedback_parts)[:6000],
            "per_shape": str(per_shape)[:6000],
        },
    )
'''


_PAIRWISE_SYSTEM_MESSAGE = """\
You are speeding up a Python numerical kernel. The reference implementation
inside the EVOLVE-BLOCK is a correct-but-very-slow triple nested loop. Your
job is to replace it with a much faster numpy / scipy / numba implementation
that returns bitwise-close results on the same inputs.

## Task
Implement `pairwise_sq_dist(X)` for a 2-D float64 array X of shape (N, D).
Return the (N, N) matrix D where `D[i, j] == sum_k (X[i, k] - X[j, k])**2`.

## Strategies (roughly in order of speedup)
1. **Gram matrix trick.** D = ||X||^2 + ||X||^2.T - 2 X @ X.T, where
   ||X||^2 is a length-N vector of per-row squared norms. ~100-1000x
   faster than the triple loop for small N, small D.
2. **`scipy.spatial.distance.cdist(X, X, 'sqeuclidean')`.** Basically
   free to write, usually matches or beats the Gram trick.
3. **`numba.njit` on the triple loop.** Great speedup per line of code,
   but numba has JIT overhead on the first call — the evaluator does a
   warmup call so this is fine.
4. **Avoid Python loops entirely.** Even a single `for i in range(N)`
   with numpy broadcasting can be 20-100x slower than the fully
   vectorised form for small inputs (function-call overhead dominates).

## Correctness requirements
- The output must have shape `(N, N)` and dtype `float64` (or a dtype
  that compares equal under `np.allclose(..., rtol=1e-6, atol=1e-6)`).
- `np.allclose` is used with `equal_nan=False`, so NaNs fail the check
  even if both sides contain them. Clip tiny negative values produced
  by `X @ X.T` rounding to 0 to avoid them turning into NaN downstream.

## Response format
OpenEvolve will apply SEARCH/REPLACE diffs to the EVOLVE-BLOCK or
accept a full rewrite. Keep the function signature
`pairwise_sq_dist(X: np.ndarray) -> np.ndarray` exactly.
"""


def make_task(problem_id: str) -> TaskSpec:
    if problem_id == "pairwise_distance":
        return TaskSpec(
            task_family="algotune",
            problem_id=problem_id,
            initial_program=_PAIRWISE_INITIAL,
            evaluator=_PAIRWISE_EVALUATOR,
            system_message=_PAIRWISE_SYSTEM_MESSAGE,
            extra_packages=["scipy>=1.11", "numba>=0.60"],
            evaluator_timeout=180,
            uses_vllm_in_evaluator=False,
        )
    raise ValueError(
        f"algotune only supports 'pairwise_distance' for now; got {problem_id!r}"
    )


FAMILY = TaskFamily(
    name="algotune",
    make_task=make_task,
    available_problems=["pairwise_distance"],
)

"""AlphaEvolve-style math task family — circle packing in a unit square.

Based on the well-known AlphaEvolve result for n=26: pack 26 non-overlapping
circles inside the unit square to maximize the sum of their radii. The best
known constant (AlphaEvolve, 2024) is ~2.635. A trivially-correct ring-based
constructor scores ~2.0-2.2, so there's meaningful headroom for OpenEvolve
to discover better arrangements.

Crashes are very rare on this task — the only ways to fail are NaN
coordinates, negative radii, overlapping circles, or circles poking out of
the unit square. Every candidate either is valid or has a clean
validation error. This makes it an excellent low-noise OE benchmark.
"""

from __future__ import annotations

from .base import TaskFamily, TaskSpec

_N_CIRCLES = 26
_ALPHAEVOLVE_TARGET = 2.635


def _initial_program() -> str:
    return '''\
"""Constructor-based circle packing for n=26 (AlphaEvolve benchmark).

Goal: define a function `construct_packing()` that returns
`(centers, radii, sum_of_radii)` for 26 non-overlapping circles inside
the unit square (axis-aligned, corners at (0,0) and (1,1)). Maximize
the sum of radii. The AlphaEvolve paper reports a best of ~2.635.

You may edit everything inside the EVOLVE-BLOCK freely: you can replace
the constructor with an optimisation loop (BFGS, simulated annealing,
analytic lattice construction, etc.) as long as `construct_packing`
returns valid numpy arrays with the right shapes and
`sum_of_radii == radii.sum()`.
"""

# EVOLVE-BLOCK-START
import numpy as np


def construct_packing():
    """Return (centers, radii, sum_of_radii) for 26 circles in [0,1]^2.

    Current placeholder: a nested ring pattern — correct but far from
    optimal. Evolution should improve the arrangement.
    """
    n = 26
    centers = np.zeros((n, 2))

    centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
    for i in range(16):
        angle = 2 * np.pi * i / 16 + np.pi / 16
        centers[i + 9] = [0.5 + 0.42 * np.cos(angle), 0.5 + 0.42 * np.sin(angle)]

    centers = np.clip(centers, 0.0, 1.0)
    radii = _max_radii(centers)
    return centers, radii, float(radii.sum())


def _max_radii(centers):
    """Largest radii that keep the circles inside the unit square and
    non-overlapping, given fixed centers. Not the tightest possible
    assignment but fast and gradient-free."""
    n = centers.shape[0]
    radii = np.array(
        [min(c[0], c[1], 1 - c[0], 1 - c[1]) for c in centers], dtype=np.float64
    )
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(((centers[i] - centers[j]) ** 2).sum()))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return np.clip(radii, 0.0, None)
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()
'''


_EVALUATOR_SOURCE = '''\
"""OpenEvolve evaluator for circle packing, n=__N__."""
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

TARGET = __TARGET__
N = __N__


def _validate(centers, radii):
    if centers.shape != (N, 2) or radii.shape != (N,):
        return False, f"bad shapes centers={centers.shape} radii={radii.shape}"
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN in centers or radii"
    if (radii < -1e-9).any():
        return False, "negative radius"
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False, f"circle {i} leaves unit square"
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.sqrt(((centers[i] - centers[j]) ** 2).sum()))
            if d + 1e-6 < radii[i] + radii[j]:
                return False, (
                    f"circles {i}/{j} overlap: dist={d:.6g}, "
                    f"r1+r2={radii[i] + radii[j]:.6g}"
                )
    return True, ""


def _run(program_path, timeout_s=90):
    out_path = program_path + ".oeout.pkl"
    runner = (
        "import pickle, sys, traceback, importlib.util, numpy as np\\n"
        "spec = importlib.util.spec_from_file_location('prog', " + repr(program_path) + ")\\n"
        "mod = importlib.util.module_from_spec(spec)\\n"
        "try:\\n"
        "    spec.loader.exec_module(mod)\\n"
        "    centers, radii, reported = mod.run_packing()\\n"
        "    pickle.dump({'centers': np.asarray(centers), 'radii': np.asarray(radii), 'reported': float(reported)}, open(" + repr(out_path) + ", 'wb'))\\n"
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
        for p in (script, out_path):
            if os.path.exists(p):
                os.unlink(p)


def evaluate(program_path):
    t0 = time.time()
    result = _run(program_path, timeout_s=90)
    elapsed = time.time() - t0

    if "error" in result:
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": float(elapsed),
                "combined_score": 0.0,
            },
            artifacts={"feedback": str(result)[:6000]},
        )

    centers = np.asarray(result["centers"], dtype=np.float64)
    radii = np.asarray(result["radii"], dtype=np.float64)
    reported = float(result.get("reported", radii.sum()))

    valid, reason = _validate(centers, radii)
    sum_radii = float(radii.sum()) if valid else 0.0
    target_ratio = sum_radii / TARGET if valid else 0.0

    return EvaluationResult(
        metrics={
            "sum_radii": sum_radii,
            "target_ratio": float(target_ratio),
            "validity": 1.0 if valid else 0.0,
            "reported_sum": reported,
            "eval_time": float(elapsed),
            "combined_score": float(target_ratio),
        },
        artifacts={
            "feedback": (
                f"valid={valid} reason={reason!r} sum_radii={sum_radii:.6f} "
                f"reported={reported:.6f} target={TARGET} "
                f"ratio={target_ratio:.4f} time={elapsed:.2f}s"
            )[:6000],
        },
    )
'''


def _evaluator() -> str:
    return _EVALUATOR_SOURCE.replace("__N__", str(_N_CIRCLES)).replace(
        "__TARGET__", repr(_ALPHAEVOLVE_TARGET)
    )


def _system_message() -> str:
    return f"""\
You are optimizing a numerical construction problem in Python.

## Goal
Pack {_N_CIRCLES} non-overlapping circles inside the unit square
[0, 1] x [0, 1] so that the sum of their radii is as large as possible.

The AlphaEvolve paper reports a best-known value of ~{_ALPHAEVOLVE_TARGET}
for n={_N_CIRCLES}. A trivial ring-based constructor achieves ~2.0-2.2.

## Output format
The EVOLVE-BLOCK must expose a function `construct_packing()` that
returns `(centers, radii, sum_of_radii)` where:
  - `centers` is a numpy array of shape ({_N_CIRCLES}, 2) with (x, y) coords
  - `radii` is a numpy array of shape ({_N_CIRCLES},) with non-negative radii
  - `sum_of_radii == radii.sum()`

All circles must lie inside [0, 1]^2 (so x-r >= 0, x+r <= 1, etc.) and
no two may overlap (pairwise distance >= r_i + r_j within 1e-6).

## Strategies that close the gap to ~2.635
1. **Analytic patterns first.** Concentric rings don't work well for
   n=26 specifically — look at honeycomb / hexagonal packings with one
   large center circle and graded rings of smaller circles near corners.
2. **Post-constructor optimisation.** Once you have a plausible
   configuration, run a gradient/penalty optimiser (scipy.optimize.minimize
   with method='L-BFGS-B' on the objective -sum(radii) plus soft overlap
   penalties). You are allowed to import scipy inside the EVOLVE-BLOCK.
3. **Repeated random restarts + local search.** Often beats a pure
   analytic construction when n doesn't admit a neat symmetric solution.
4. **Mixed approach.** Hand-placed seeds (corners, center, midpoints of
   edges) followed by a numerical tightening step.

## Response format
OpenEvolve will apply SEARCH/REPLACE diffs to the EVOLVE-BLOCK, OR you
can replace the entire block with a full rewrite. Either is accepted.
The evaluator validates shapes + non-overlap + square containment; a
candidate that violates any of these scores 0.

Be careful with numpy array shapes: `centers.shape == ({_N_CIRCLES}, 2)`
exactly (not (2, {_N_CIRCLES}) or ({_N_CIRCLES},) or anything else).
"""


def make_task(problem_id: str) -> TaskSpec:
    assert problem_id == "circle_packing_26", (
        f"alphaevolve only supports 'circle_packing_26'; got {problem_id!r}"
    )
    return TaskSpec(
        task_family="alphaevolve",
        problem_id=problem_id,
        initial_program=_initial_program(),
        evaluator=_evaluator(),
        system_message=_system_message(),
        extra_packages=["scipy>=1.11"],
        evaluator_timeout=120,
        uses_vllm_in_evaluator=False,
    )


FAMILY = TaskFamily(
    name="alphaevolve",
    make_task=make_task,
    available_problems=["circle_packing_26"],
)

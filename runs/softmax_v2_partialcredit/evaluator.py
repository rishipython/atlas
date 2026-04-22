"""Auto-generated OpenEvolve evaluator for problem_id='softmax'."""
from __future__ import annotations

import sys
sys.path.insert(0, "/atlas")

from benchmark.kernel.evaluator import evaluate_kernel
from benchmark.kernel.problems import KERNEL_PROBLEMS
from openevolve.evaluation_result import EvaluationResult

PROBLEM_ID = 'softmax'


def _per_shape_score(r: dict) -> float:
    """Graduated per-shape credit. See module docstring for levels."""
    if r.get("correct"):
        speedup = float(r.get("speedup", 0.0))
        return 0.5 + 0.5 * min(speedup, 2.0)
    # "error" field is populated when the call raised — compile error,
    # wrong-shape output, pointer issue, etc.  This is a runnable but
    # broken attempt and is very useful context for the LLM.
    if "error" in r:
        return 0.05
    # No error string + not correct means the call returned a tensor
    # whose values didn't match the reference — closer to "almost right"
    # than a full crash, so it gets a slightly higher partial-credit score.
    return 0.15


def evaluate(program_path: str):
    with open(program_path, "r") as f:
        code = f.read()

    problem = KERNEL_PROBLEMS[PROBLEM_ID]
    result = evaluate_kernel(problem, code, timeout=180)

    per_shape = (result.metadata or {}).get("per_shape", [])
    if per_shape:
        shape_scores = [_per_shape_score(r) for r in per_shape]
        progress = sum(shape_scores) / len(shape_scores)
    else:
        # Subprocess crashed before printing any per-shape result. Leave
        # progress at 0 — feedback still carries the traceback as an
        # artifact so the LLM can see what went wrong.
        shape_scores = []
        progress = 0.0

    metrics = {
        "correctness": 1.0 if result.correct else 0.0,
        "speedup": float(result.score),  # strict: 0 unless every shape correct
        "combined_score": float(progress),
    }

    artifacts = {
        "feedback": result.feedback[:6000],
        "per_shape": str(per_shape)[:6000],
        "shape_scores": str(shape_scores),
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)

"""Auto-generated OpenEvolve evaluator for problem_id='softmax'."""
from __future__ import annotations

import sys
sys.path.insert(0, "/atlas")

from benchmark.kernel.evaluator import evaluate_kernel
from benchmark.kernel.problems import KERNEL_PROBLEMS
from openevolve.evaluation_result import EvaluationResult

PROBLEM_ID = 'softmax'


def evaluate(program_path: str):
    with open(program_path, "r") as f:
        code = f.read()

    problem = KERNEL_PROBLEMS[PROBLEM_ID]
    result = evaluate_kernel(problem, code, timeout=180)

    metrics = {
        "correctness": 1.0 if result.correct else 0.0,
        "speedup": float(result.score),
        # OpenEvolve uses combined_score as the fitness signal. We want
        # correctness to be a hard prerequisite, so we set it to the
        # speedup (which evaluate_kernel already zeros out on failure).
        "combined_score": float(result.score),
    }

    per_shape = (result.metadata or {}).get("per_shape", [])
    artifacts = {
        "feedback": result.feedback[:4000],
        "per_shape": str(per_shape)[:4000],
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)

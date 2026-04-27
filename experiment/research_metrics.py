from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Any


DEFAULT_PASS_KS = [1, 5, 10, 20, 50, 100]


def parse_pass_ks(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_PASS_KS)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(k for k in out if k > 0))


def estimate_pass_at_k(num_total: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimate from Chen et al. / HumanEval.

    Given ``num_total`` draws and ``num_correct`` correct samples, estimate the
    probability that at least one of ``k`` fresh samples is correct.
    """
    if k <= 0 or num_total <= 0:
        return 0.0
    k = min(k, num_total)
    c = max(0, min(num_correct, num_total))
    if c == 0:
        return 0.0
    if num_total - c < k:
        return 1.0
    return 1.0 - (comb(num_total - c, k) / comb(num_total, k))


def summarize_sample_records(
    samples: list[dict[str, Any]],
    *,
    pass_ks: list[int] | None = None,
) -> dict[str, Any]:
    pass_ks = sorted(set(pass_ks or DEFAULT_PASS_KS))
    n = len(samples)
    correct_samples = [s for s in samples if s.get("correct")]
    c = len(correct_samples)
    mean_speedup = (
        sum(float(s.get("score", 0.0)) for s in correct_samples) / c if c else 0.0
    )
    best_speedup = max((float(s.get("score", 0.0)) for s in correct_samples), default=0.0)
    correctness_rate = c / n if n else 0.0
    expected_speedup = correctness_rate * mean_speedup

    first_correct_index = None
    for idx, sample in enumerate(samples, start=1):
        if sample.get("correct"):
            first_correct_index = idx
            break

    return {
        "n_samples": n,
        "num_correct": c,
        "correctness_rate": correctness_rate,
        "mean_speedup_when_correct": mean_speedup,
        "expected_speedup": expected_speedup,
        "best_speedup": best_speedup,
        "pass_at_1": estimate_pass_at_k(n, c, 1),
        "pass_at_k": estimate_pass_at_k(n, c, n) if n else 0.0,
        "pass_at_ks": {
            str(k): estimate_pass_at_k(n, c, k)
            for k in pass_ks
            if k <= n
        },
        "first_correct_sample": first_correct_index,
    }


def summarize_search_records(
    trace_records: list[dict[str, Any]],
    *,
    pass_ks: list[int] | None = None,
) -> dict[str, Any]:
    pass_ks = sorted(set(pass_ks or DEFAULT_PASS_KS))
    candidates = []
    best_score_so_far = 0.0
    best_speedup_so_far = 0.0
    first_correct_iter = None

    for rec in trace_records:
        metrics = rec.get("child_metrics") or {}
        correct = float(metrics.get("correctness", 0.0)) >= 1.0
        score = float(metrics.get("combined_score", 0.0))
        speedup = float(metrics.get("speedup", 0.0)) if correct else 0.0
        best_score_so_far = max(best_score_so_far, score)
        best_speedup_so_far = max(best_speedup_so_far, speedup)
        iteration = int(rec.get("iteration", len(candidates) + 1))
        if correct and first_correct_iter is None:
            first_correct_iter = iteration
        candidates.append(
            {
                "iteration": iteration,
                "correct": correct,
                "score": score,
                "speedup": speedup,
                "best_score_so_far": best_score_so_far,
                "best_speedup_so_far": best_speedup_so_far,
            }
        )

    sample_like = [{"correct": c["correct"], "score": c["speedup"]} for c in candidates]
    summary = summarize_sample_records(sample_like, pass_ks=pass_ks)
    summary["num_iterations"] = len(candidates)
    summary["first_correct_iteration"] = first_correct_iter
    summary["best_combined_score"] = best_score_so_far
    summary["best_speedup_so_far_curve"] = [c["best_speedup_so_far"] for c in candidates]
    summary["best_combined_score_curve"] = [c["best_score_so_far"] for c in candidates]
    return {"summary": summary, "candidates": candidates}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())

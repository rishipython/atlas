"""Build an AlgoTune SFT dataset from search or Best-of-N outputs.

This branch uses a single training format:

- prompt: the short base AlgoTune prompt used at evaluation time
- target: a full runnable Python file
- weight: one scalar per example for reward-aware training

The builder supports two data sources:

1. OpenEvolve traces via ``--trace``
2. Standalone sampling directories via ``--sample-dir``

The main ablations map to two simple knobs:

- ``--selection all`` vs ``--selection best_per_problem``
- ``--weight-scheme reward`` vs ``correct_only`` vs ``uniform``
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment.algotune_prompts import build_algotune_prompts

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_num, path, exc)
    return rows


def _problem_id_from_trace_path(path: Path) -> str | None:
    parts = path.parts
    if "runs" not in parts:
        return None
    try:
        run_dir = parts[parts.index("runs") + 1]
    except IndexError:
        return None
    return run_dir.split("_", 1)[0]


def _trace_examples(path: Path) -> list[dict[str, Any]]:
    problem_id = _problem_id_from_trace_path(path)
    rows = _iter_jsonl(path)
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        child_code = (row.get("child_code") or "").strip()
        metrics = row.get("child_metrics") or {}
        reward = float(metrics.get("combined_score", 0.0) or 0.0)
        correct = float(metrics.get("correctness", 0.0) or 0.0) >= 1.0
        if not child_code:
            continue
        examples.append(
            {
                "problem_id": row.get("problem_id") or problem_id,
                "code": child_code + "\n",
                "reward": reward,
                "correct": correct,
                "source": "openevolve",
                "source_path": str(path),
                "source_idx": idx,
            }
        )
    return examples


def _sample_dir_examples(path: Path) -> list[dict[str, Any]]:
    samples_path = path / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Expected {samples_path}")

    problem_id = path.name.split("__", 1)[0]
    rows = json.loads(samples_path.read_text())
    examples: list[dict[str, Any]] = []
    for row in rows:
        sample_idx = int(row["sample_idx"])
        code_path = path / f"sample_{sample_idx:03d}_code.py"
        code = code_path.read_text() if code_path.exists() else ""
        if not code.strip():
            continue
        examples.append(
            {
                "problem_id": problem_id,
                "code": code.rstrip() + "\n",
                "reward": float(row.get("combined_score", 0.0) or 0.0),
                "correct": bool(row.get("correct", False)),
                "source": "best_of_n",
                "source_path": str(path),
                "source_idx": sample_idx,
            }
        )
    return examples


def _compute_weight(example: dict[str, Any], scheme: str) -> float:
    if scheme == "reward":
        return max(0.0, float(example["reward"]))
    if scheme == "correct_only":
        return 1.0 if example["correct"] else 0.0
    if scheme == "uniform":
        return 1.0
    raise ValueError(f"Unknown weight scheme: {scheme}")


def _filter_examples(
    examples: list[dict[str, Any]],
    *,
    allowed_problems: set[str] | None,
    min_reward: float,
    require_correct: bool,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for example in examples:
        problem_id = example.get("problem_id")
        if not problem_id:
            continue
        if allowed_problems and problem_id not in allowed_problems:
            continue
        if float(example["reward"]) < min_reward:
            continue
        if require_correct and not example["correct"]:
            continue
        kept.append(example)
    return kept


def _select_examples(examples: list[dict[str, Any]], selection: str) -> list[dict[str, Any]]:
    if selection == "all":
        return examples
    if selection != "best_per_problem":
        raise ValueError(f"Unknown selection mode: {selection}")

    best: dict[str, dict[str, Any]] = {}
    for example in examples:
        problem_id = example["problem_id"]
        incumbent = best.get(problem_id)
        if incumbent is None or float(example["reward"]) > float(incumbent["reward"]):
            best[problem_id] = example
    return [best[key] for key in sorted(best)]


def _to_record(example: dict[str, Any], weight_scheme: str) -> dict[str, Any]:
    system_msg, user_msg, _ = build_algotune_prompts(example["problem_id"])
    return {
        "problem_id": example["problem_id"],
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["code"]},
        ],
        "sample_weight": _compute_weight(example, weight_scheme),
        "reward": float(example["reward"]),
        "correct": bool(example["correct"]),
        "source": example["source"],
        "source_path": example["source_path"],
        "source_idx": example["source_idx"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", default=[], help="Path to an OpenEvolve evolution_trace.jsonl file.")
    parser.add_argument("--sample-dir", action="append", default=[], help="Path to an eval directory containing samples.json.")
    parser.add_argument("--problems", default="", help="Comma-separated subset of problem ids to keep.")
    parser.add_argument("--selection", choices=["all", "best_per_problem"], default="all")
    parser.add_argument("--weight-scheme", choices=["reward", "correct_only", "uniform"], default="reward")
    parser.add_argument("--min-reward", type=float, default=0.0)
    parser.add_argument("--require-correct", action="store_true", help="Keep only correct samples before selection.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    all_examples: list[dict[str, Any]] = []
    for raw_path in args.trace:
        all_examples.extend(_trace_examples(Path(raw_path)))
    for raw_path in args.sample_dir:
        all_examples.extend(_sample_dir_examples(Path(raw_path)))

    allowed_problems = {
        problem.strip() for problem in args.problems.split(",") if problem.strip()
    } or None
    filtered = _filter_examples(
        all_examples,
        allowed_problems=allowed_problems,
        min_reward=args.min_reward,
        require_correct=args.require_correct,
    )
    selected = _select_examples(filtered, args.selection)

    if args.weight_scheme == "correct_only":
        selected = [row for row in selected if row["correct"]]

    records = [_to_record(example, args.weight_scheme) for example in selected]
    records = [record for record in records if record["sample_weight"] > 0.0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_problem: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    for record in records:
        by_problem[record["problem_id"]] += 1
        by_source[record["source"]] += 1

    logger.info("Wrote %d records to %s", len(records), out_path)
    logger.info("Problems: %s", dict(sorted(by_problem.items())))
    logger.info("Sources: %s", dict(sorted(by_source.items())))


if __name__ == "__main__":
    main()

"""Summarize OpenEvolve runs as search-efficiency curves.

This script reads one or more ``evolution_trace.jsonl`` files, computes the
same correctness / speed metrics used by standalone sampling, and writes a
compact JSON summary plus optional CSV-like console table.  It is the glue for
comparing ``Base + OpenEvolve`` vs ``ATLAS + OpenEvolve``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment.research_metrics import parse_pass_ks, summarize_search_records


def _infer_problem_id(path: Path) -> str:
    parts = path.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1].split("_", 1)[0]
    return path.parent.parent.parent.name.split("_", 1)[0] or "unknown"


def _infer_tag(path: Path) -> str:
    run_dir = path.parent.parent.parent.name
    if "atlas" in run_dir:
        return "atlas_search"
    return "base_search"


def _load_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--pass-ks",
        default="1,5,10,20,50,100",
        help="Comma-separated pass@k cutoffs to report.",
    )
    args = parser.parse_args()

    pass_ks = parse_pass_ks(args.pass_ks)
    summaries = []
    for raw_path in args.traces:
        path = Path(raw_path)
        records = _load_trace(path)
        payload = summarize_search_records(records, pass_ks=pass_ks)
        summary = payload["summary"]
        summary.update(
            {
                "trace_path": str(path),
                "problem_id": _infer_problem_id(path),
                "tag": _infer_tag(path),
            }
        )
        summaries.append({"summary": summary, "candidates": payload["candidates"]})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": summaries}, indent=2))

    print(
        f"{'problem':<12} {'tag':<14} {'iters':>6} {'pass@1':>7} {'pass@k':>7} "
        f"{'exp_sp':>8} {'best_sp':>8} {'first_ok':>9}"
    )
    for item in summaries:
        s = item["summary"]
        print(
            f"{s['problem_id']:<12} {s['tag']:<14} {s['num_iterations']:>6} "
            f"{s['pass_at_1']:>7.2f} {s['pass_at_k']:>7.2f} "
            f"{s['expected_speedup']:>8.3f} {s['best_speedup']:>8.3f} "
            f"{str(s['first_correct_iteration']):>9}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

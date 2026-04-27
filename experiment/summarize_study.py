"""Aggregate standalone and search summaries into a single study report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", nargs="*", default=[])
    parser.add_argument("--search", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    standalone_rows = []
    for path in args.standalone:
        payload = _load(path)
        for leg in payload.get("legs", []):
            standalone_rows.append(
                {
                    "source": path,
                    "setting": leg.get("tag"),
                    "problem_id": leg.get("problem_id"),
                    "pass_at_1": leg.get("pass_at_1"),
                    "pass_at_k": leg.get("pass_at_k"),
                    "pass_at_ks": leg.get("pass_at_ks", {}),
                    "correctness_rate": leg.get("correctness_rate"),
                    "expected_speedup": leg.get("expected_speedup"),
                    "mean_speedup_when_correct": leg.get("mean_speedup_when_correct"),
                    "best_speedup": leg.get("best_speedup_when_correct", leg.get("best_speedup")),
                    "first_correct_sample": leg.get("first_correct_sample"),
                }
            )

    search_rows = []
    if args.search:
        payload = _load(args.search)
        for item in payload.get("runs", []):
            s = item.get("summary", {})
            search_rows.append(
                {
                    "source": args.search,
                    "setting": s.get("tag"),
                    "problem_id": s.get("problem_id"),
                    "pass_at_1": s.get("pass_at_1"),
                    "pass_at_k": s.get("pass_at_k"),
                    "pass_at_ks": s.get("pass_at_ks", {}),
                    "correctness_rate": s.get("correctness_rate"),
                    "expected_speedup": s.get("expected_speedup"),
                    "mean_speedup_when_correct": s.get("mean_speedup_when_correct"),
                    "best_speedup": s.get("best_speedup"),
                    "first_correct_iteration": s.get("first_correct_iteration"),
                }
            )

    report = {
        "standalone": standalone_rows,
        "search": search_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"{'setting':<16} {'problem':<12} {'pass@1':>7} {'pass@k':>7} {'exp_sp':>8} {'best_sp':>8}")
    for row in standalone_rows + search_rows:
        print(
            f"{str(row.get('setting')):<16} {row['problem_id']:<12} "
            f"{row['pass_at_1']:>7.2f} {row['pass_at_k']:>7.2f} "
            f"{row['expected_speedup']:>8.3f} {row['best_speedup']:>8.3f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

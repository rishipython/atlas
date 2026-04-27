"""Build two-phase advantage-weighted SFT datasets from the synth advantage file.

Phase 1  (correctness):   sample_weight = +1 for CORRECT, -1 for WRONG_VALUES/CRASHED.
Phase 2  (speedup):       only CORRECT records, sample_weight = z-score of
                          combined_score within the correct group (which is
                          monotonic in mean-speedup for the correct class,
                          since score = 0.5 + 0.5*mean_speedup when correct).

Both output files are drop-in compatible with ``train_atlas_sft.py`` (which
reads ``messages`` + ``sample_weight``).

Usage::

    python experiment/build_twophase_sft.py \
        --in data/synth_softmax_all_advantage_v1.jsonl \
        --out-p1 data/synth_softmax_twophase_p1.jsonl \
        --out-p2 data/synth_softmax_twophase_p2.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out-p1", required=True)
    p.add_argument("--out-p2", required=True)
    args = p.parse_args()

    recs = [json.loads(l) for l in Path(args.in_path).read_text().splitlines() if l.strip()]
    correct = [r for r in recs if r["classification"] == "correct"]
    bad = [r for r in recs if r["classification"] != "correct"]

    print(
        f"[twophase] source: n={len(recs)}  correct={len(correct)}  bad={len(bad)}"
    )

    # --- Phase 1: all records, weight = +/- 1 by correctness ---
    p1_records: list[dict] = []
    for r in recs:
        weight = 1.0 if r["classification"] == "correct" else -1.0
        out = {
            "problem_id": r["problem_id"],
            "phase": "p1_correctness",
            "sample_weight": weight,
            "combined_score": r["combined_score"],
            "classification": r["classification"],
            "messages": r["messages"],
            "_meta": {**r.get("_meta", {}), "phase_weight": weight},
        }
        p1_records.append(out)

    # --- Phase 2: correct-only, z-score of combined_score ---
    scores = [r["combined_score"] for r in correct]
    mean_c = statistics.mean(scores)
    std_c = statistics.stdev(scores) if len(scores) > 1 else 1.0
    std_c = std_c or 1e-6

    p2_records: list[dict] = []
    for r in correct:
        z = (r["combined_score"] - mean_c) / std_c
        out = {
            "problem_id": r["problem_id"],
            "phase": "p2_speedup",
            "sample_weight": float(z),
            "combined_score": r["combined_score"],
            "classification": r["classification"],
            "messages": r["messages"],
            "_meta": {**r.get("_meta", {}), "z_score": z},
        }
        p2_records.append(out)

    p1_path = Path(args.out_p1)
    p2_path = Path(args.out_p2)
    p1_path.parent.mkdir(parents=True, exist_ok=True)
    p2_path.parent.mkdir(parents=True, exist_ok=True)
    with p1_path.open("w") as f:
        for r in p1_records:
            f.write(json.dumps(r) + "\n")
    with p2_path.open("w") as f:
        for r in p2_records:
            f.write(json.dumps(r) + "\n")

    p1_wts = [r["sample_weight"] for r in p1_records]
    p2_wts = [r["sample_weight"] for r in p2_records]
    print(f"[twophase] wrote {len(p1_records)} phase-1 records → {p1_path}")
    print(f"           weights: +1 count={sum(1 for w in p1_wts if w > 0)}  "
          f"-1 count={sum(1 for w in p1_wts if w < 0)}")
    print(f"[twophase] wrote {len(p2_records)} phase-2 records → {p2_path}")
    if p2_wts:
        print(
            f"           z-score range=[{min(p2_wts):+.3f}, {max(p2_wts):+.3f}]  "
            f"mean={sum(p2_wts) / len(p2_wts):+.3f}  "
            f"(source mean_score={mean_c:.4f}, std={std_c:.4f})"
        )


if __name__ == "__main__":
    main()

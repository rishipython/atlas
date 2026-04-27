"""Summarise and plot pass@40 results for algotune problems.

Reads the per-sample data produced by ``eval_algotune.py`` for the
base and ATLAS legs across three problems and writes:
  - a single markdown table with mean / std / best speedup (correct-only),
  - per-problem histograms comparing base vs ATLAS speedup distributions.

Usage (local; no Modal):
    python experiment/plot_algotune_pass40.py \
        --run-dir eval_runs/atlas_algotune_pass40_v1/atlas_algotune_pass40_v1 \
        --out-dir eval_runs/atlas_algotune_pass40_v1/analysis
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROBLEMS = ["fft_convolution", "convolve2d_full_fill", "affine_transform_2d"]
LEGS = ["base", "atlas"]


def _load(run_dir: Path, problem: str, leg: str) -> list[dict]:
    p = run_dir / f"{problem}__{leg}" / "samples.json"
    return json.loads(p.read_text())


def _bootstrap_pass_at_k(speedups: list[float], k: int, n_boot: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    """Return (mean, lo95, hi95) of the expected maximum speedup across k draws.

    ``speedups`` is the list of per-sample speedups (0 for incorrect).
    We sample with replacement ``k`` values and take their max; the
    bootstrap replicates this ``n_boot`` times to get a confidence band.
    """
    import random

    rng = random.Random(seed)
    draws = []
    n = len(speedups)
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(k)]
        draws.append(max(speedups[i] for i in idxs))
    draws.sort()
    return (
        sum(draws) / len(draws),
        draws[int(0.025 * len(draws))],
        draws[int(0.975 * len(draws))],
    )


def _load_oe_trajectories(trace_path: Path) -> list[dict]:
    """Return the OE evolution trace's per-record speedup / correct list.

    Each record is coerced into the same shape the pass@k pipeline uses
    (``correct`` + ``speedup``) so it flows through the downstream
    aggregation code unchanged.
    """
    out: list[dict] = []
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r.get("child_metrics") or {}
        out.append(
            {
                "correct": float(m.get("correctness", 0.0)) >= 0.99,
                "speedup": float(m.get("speedup", 0.0)),
                "combined_score": float(m.get("combined_score", 0.0)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir with <problem>__<leg>/samples.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--oe-trace",
        action="append",
        default=[],
        help="optional OE evolution_trace.jsonl to include as an extra 'oe' "
        "row. Format: problem_id=path/to/evolution_trace.jsonl. Can repeat.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse --oe-trace mapping: problem_id -> Path
    oe_traces: dict[str, Path] = {}
    for spec in args.oe_trace:
        if "=" not in spec:
            raise SystemExit(f"--oe-trace expects problem_id=path, got {spec!r}")
        pid, p = spec.split("=", 1)
        oe_traces[pid.strip()] = Path(p.strip())

    # ---- Table --------------------------------------------------------
    rows = []
    for prob in PROBLEMS:
        # Add OE row first (if provided for this problem) so it visually
        # anchors the comparison with the same-problem base/atlas legs.
        if prob in oe_traces:
            oe_samples = _load_oe_trajectories(oe_traces[prob])
            n = len(oe_samples)
            correct = [s for s in oe_samples if s["correct"]]
            speedups_correct = [s["speedup"] for s in correct]
            speedups_all = [s["speedup"] if s["correct"] else 0.0 for s in oe_samples]
            best = max(speedups_correct, default=0.0)
            mean_c = statistics.mean(speedups_correct) if speedups_correct else 0.0
            std_c = statistics.stdev(speedups_correct) if len(speedups_correct) > 1 else 0.0
            pass_k = len(correct) / max(1, n)
            boot_max_mean, lo, hi = _bootstrap_pass_at_k(speedups_all, n)
            rows.append(
                {
                    "problem": prob,
                    "leg": "oe",
                    "n": n,
                    "n_correct": len(correct),
                    "pass@k": pass_k,
                    "best_speedup": best,
                    "mean_speedup_correct": mean_c,
                    "std_speedup_correct": std_c,
                    "exp_max_mean": boot_max_mean,
                    "exp_max_lo95": lo,
                    "exp_max_hi95": hi,
                }
            )
        for leg in LEGS:
            samples = _load(run_dir, prob, leg)
            n = len(samples)
            correct = [s for s in samples if s.get("correct")]
            speedups_correct = [float(s["speedup"]) for s in correct]
            speedups_all = [float(s["speedup"]) if s.get("correct") else 0.0 for s in samples]
            best = max(speedups_correct, default=0.0)
            mean_c = statistics.mean(speedups_correct) if speedups_correct else 0.0
            std_c = statistics.stdev(speedups_correct) if len(speedups_correct) > 1 else 0.0
            pass_k = len(correct) / max(1, n)
            boot_max_mean, lo, hi = _bootstrap_pass_at_k(speedups_all, n)
            rows.append(
                {
                    "problem": prob,
                    "leg": leg,
                    "n": n,
                    "n_correct": len(correct),
                    "pass@k": pass_k,
                    "best_speedup": best,
                    "mean_speedup_correct": mean_c,
                    "std_speedup_correct": std_c,
                    "exp_max_mean": boot_max_mean,
                    "exp_max_lo95": lo,
                    "exp_max_hi95": hi,
                }
            )

    # Markdown table. n differs between legs (OE may have run with a
    # different budget than pass@40), so we show the per-row budget in
    # the "correct/n" column rather than baking it into the header.
    md_lines = [
        "| problem | leg | correct/n | pass rate | best | mean±std (correct) | E[max@n] (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['problem']} | **{r['leg']}** | {r['n_correct']}/{r['n']} | "
            f"{r['pass@k']:.2%} | {r['best_speedup']:.2f}× | "
            f"{r['mean_speedup_correct']:.2f}±{r['std_speedup_correct']:.2f}× | "
            f"{r['exp_max_mean']:.2f}× "
            f"[{r['exp_max_lo95']:.2f}, {r['exp_max_hi95']:.2f}] |"
        )
    md_text = "\n".join(md_lines)
    (out_dir / "summary_table.md").write_text(md_text + "\n")
    print(md_text)

    # JSON dump for downstream consumers
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))

    # ---- Histograms ---------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping histograms")
        return

    for prob in PROBLEMS:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
        colors = {"base": "#7f8fa6", "atlas": "#c0392b"}
        for leg in LEGS:
            samples = _load(run_dir, prob, leg)
            speedups = [float(s["speedup"]) for s in samples if s.get("correct")]
            if not speedups:
                continue
            ax.hist(
                speedups,
                bins=20,
                alpha=0.55,
                label=f"{leg} (n_correct={len(speedups)})",
                color=colors.get(leg, "#333"),
                edgecolor="white",
            )
        ax.set_xlabel("speedup ×")
        ax.set_ylabel("count")
        ax.set_title(f"{prob}: speedup distribution (correct samples only)")
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"speedup_hist_{prob}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()

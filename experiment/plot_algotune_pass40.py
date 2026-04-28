"""Plot reward-vs-k curves from `experiment/eval_algotune.py` output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", required=True, help="Path to compare.json")
    parser.add_argument("--out", required=True, help="Output image path")
    args = parser.parse_args()

    compare = json.loads(Path(args.compare).read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for leg in compare["legs"]:
        grouped[leg["tag"]].append(leg)

    plt.figure(figsize=(8, 5))
    for tag, legs in sorted(grouped.items()):
        n = max(len(leg["best_reward_by_k"]) for leg in legs)
        means = []
        for idx in range(n):
            values = []
            for leg in legs:
                curve = leg["best_reward_by_k"]
                if idx < len(curve):
                    values.append(curve[idx])
            means.append(sum(values) / len(values))
        xs = list(range(1, len(means) + 1))
        plt.plot(xs, means, label=tag, linewidth=2)

    plt.xlabel("Evaluated candidates k")
    plt.ylabel("Best reward so far")
    plt.title(f"AlgoTune reward-vs-k: {compare['run_name']}")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(out_path)


if __name__ == "__main__":
    main()

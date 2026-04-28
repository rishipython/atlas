"""Build DPO-style preference dataset from the synth advantage dataset.

Pairs a CORRECT trajectory (chosen) with an incorrect one (rejected) for
the same problem.  We sample incorrect trajectories from both CRASHED and
WRONG_VALUES so the policy learns to prefer runnable, correct kernels to
both broken and numerically wrong ones.

Output schema (one JSON per line, TRL-compatible):

    {
      "prompt": [system, user]            # list[dict] with role/content
      "chosen": [assistant_correct]       # list[dict] with role/content (+thinking)
      "rejected": [assistant_wrong]       # list[dict] with role/content (+thinking)
      "_meta": {...}
    }

Usage::

    python experiment/build_synth_dpo_dataset.py \
        --in data/synth_softmax_all_advantage_v1.jsonl \
        --out data/synth_softmax_dpo_v1.jsonl \
        --max-pairs-per-correct 2 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--max-pairs-per-correct",
        type=int,
        default=2,
        help="For each correct record, pair it with this many incorrect records.",
    )
    p.add_argument(
        "--wrong-fraction",
        type=float,
        default=0.5,
        help="Fraction of negatives that should be WRONG_VALUES rather than CRASHED.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    recs = [json.loads(l) for l in Path(args.in_path).read_text().splitlines() if l.strip()]
    correct = [r for r in recs if r["classification"] == "correct"]
    wrong = [r for r in recs if r["classification"] == "wrong_values"]
    crashed = [r for r in recs if r["classification"] == "crashed"]

    print(
        f"[dpo] source: correct={len(correct)} wrong_values={len(wrong)} "
        f"crashed={len(crashed)} total={len(recs)}"
    )
    assert correct and (wrong or crashed), "Need at least one correct and one incorrect."

    def _drop_thinking(msg: dict) -> dict:
        # The gpt-oss chat template applied by TRL's DPOTrainer renders
        # ``content`` directly; ``thinking`` is a custom field we add for SFT
        # training.  Keep DPO targets focused on the final code.
        return {k: v for k, v in msg.items() if k != "thinking"}

    rng = random.Random(args.seed)
    pairs: list[dict] = []
    for c in correct:
        system = c["messages"][0]
        user = c["messages"][1]
        chosen_assistant = _drop_thinking(c["messages"][2])
        for _ in range(args.max_pairs_per_correct):
            # Choose a negative: WRONG_VALUES or CRASHED.
            if wrong and (not crashed or rng.random() < args.wrong_fraction):
                neg = rng.choice(wrong)
            else:
                neg = rng.choice(crashed)
            rejected_assistant = _drop_thinking(neg["messages"][2])
            # Full-conversation format expected by train_atlas_lora.py:
            # ``chosen`` and ``rejected`` are each the complete conversation
            # (system, user, assistant), and the trainer infers the shared
            # prefix automatically.  Prompt field is left null for compat.
            margin = max(0.0, float(c["combined_score"]) - float(neg["combined_score"]))
            pairs.append(
                {
                    "prompt": None,
                    "chosen": [system, user, chosen_assistant],
                    "rejected": [system, user, rejected_assistant],
                    "chosen_score": c["combined_score"],
                    "rejected_score": neg["combined_score"],
                    "chosen_correct": True,
                    "rejected_correct": False,
                    "margin": margin,
                    "pairing_type": "correct_vs_" + neg["classification"],
                    "chosen_iter": c["_meta"]["iteration"],
                    "rejected_iter": neg["_meta"]["iteration"],
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in pairs:
            f.write(json.dumps(r) + "\n")

    n_rej_crash = sum(1 for p in pairs if p["pairing_type"].endswith("crashed"))
    n_rej_wrong = sum(1 for p in pairs if p["pairing_type"].endswith("wrong_values"))
    print(
        f"[dpo] wrote {len(pairs)} pairs → {out_path}  "
        f"(rejected: crashed={n_rej_crash}, wrong_values={n_rej_wrong})"
    )


if __name__ == "__main__":
    main()

"""Build an advantage-weighted SFT dataset from OE trajectories + synth reasoning.

For each trajectory in an OE ``evolution_trace.jsonl``:

  1. Classify the trajectory as CORRECT, WRONG_VALUES, or CRASHED.
  2. Compute an "advantage" score:
     * CORRECT:        (combined_score - mean_correct) / std_correct.
                       Z-score within the group of trajectories that ran
                       and passed every test shape.
     * WRONG_VALUES:   fixed penalty, strictly more negative than the
                       lowest CORRECT advantage.
     * CRASHED:        fixed penalty, strictly more negative than
                       WRONG_VALUES (subprocess crashed before shape
                       results were even collected — the worst outcome).
  3. Load the synthesized reasoning trace produced by
     ``synth_reasoning.main_all`` for this iteration (``iter_NNN_content.txt``)
     and clean it with ``build_synth_sft.clean_synth_trace``.
  4. Emit one SFT record with ``advantage`` in ``sample_weight`` and the
     classification in ``_meta`` for bookkeeping.

The training script (``train_atlas_sft.py``) multiplies per-example NLL
by ``sample_weight`` (advantage), so negative advantages unlearn bad
trajectories and positive advantages distill good ones.

Usage::

    python experiment/build_advantage_sft.py \
        --trace runs/softmax_v5_reasoning/oe/evolution_trace.jsonl \
        --synth-dir /tmp/synth_softmax_all_v1 \
        --out data/synth_softmax_all_advantage_v1.jsonl \
        --problem-id softmax
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiment.build_synth_sft import (  # noqa: E402
    _strip_oe_scaffold,
    clean_synth_trace,
)
from experiment.algotune_prompts import build_algotune_prompts  # noqa: E402


CORRECT = "correct"
WRONG_VALUES = "wrong_values"
CRASHED = "crashed"


def _classify(rec: dict) -> str:
    m = rec.get("child_metrics") or {}
    correctness = float(m.get("correctness", 0.0))
    combined = float(m.get("combined_score", 0.0))
    # correctness == 1.0  ⇒  every shape produced numerically close output.
    if correctness >= 1.0:
        return CORRECT
    # Full subprocess crash (no per-shape results at all) ⇒ combined is 0.0.
    # Per-shape hard-raises are 0.05 each (averaged ≤ 0.06).  Below ~0.06
    # means "every shape either hard-raised or subprocess died".
    if combined <= 0.06:
        return CRASHED
    # Otherwise the kernel ran on ≥1 shape with wrong numerical output
    # (per-shape score 0.15 for mismatch) or a mix.
    return WRONG_VALUES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="evolution_trace.jsonl path")
    parser.add_argument("--synth-dir", required=True, help="dir of synth iter_NNN_content.txt files")
    parser.add_argument("--out", required=True, help="output JSONL")
    parser.add_argument("--problem-id", required=True)
    parser.add_argument(
        "--handoff-suffix",
        default="kernel",
        help="Handoff word for clean_synth_trace ('kernel' for Triton, 'it' for algotune).",
    )
    parser.add_argument(
        "--wrong-values-gap",
        type=float,
        default=0.5,
        help="WRONG_VALUES advantage = min_correct_adv - this.",
    )
    parser.add_argument(
        "--crashed-gap",
        type=float,
        default=1.5,
        help="CRASHED advantage = min_correct_adv - this.",
    )
    parser.add_argument(
        "--adv-clip-min",
        type=float,
        default=-2.5,
        help="Lower clip bound for advantage (applied as max(adv, adv_clip_min)).",
    )
    parser.add_argument(
        "--adv-clip-max",
        default="2.5",
        help=(
            "Upper clip bound for advantage. Pass 'none' to disable upper clipping "
            "(one-sided lower-only clip)."
        ),
    )
    parser.add_argument(
        "--prompt-source",
        choices=["auto", "canonical", "synth_context"],
        default="auto",
        help=(
            "Prompt source for training records. "
            "'auto' prefers canonical current prompts when available."
        ),
    )
    parser.add_argument(
        "--strict-prompt-match",
        action="store_true",
        help=(
            "When canonical prompts are used and synth context exists, fail if "
            "stored synth prompts differ from canonical prompts."
        ),
    )
    args = parser.parse_args()

    clip_max_raw = str(args.adv_clip_max).strip().lower()
    clip_max = None if clip_max_raw in {"none", "null", "inf", "+inf"} else float(args.adv_clip_max)
    clip_min = float(args.adv_clip_min)
    if clip_max is not None and clip_min > clip_max:
        raise ValueError(f"adv-clip-min ({clip_min}) must be <= adv-clip-max ({clip_max})")

    trace_path = Path(args.trace)
    synth_dir = Path(args.synth_dir)
    out_path = Path(args.out)

    records = [
        json.loads(line)
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]

    # --- Pass 1: compute z-score statistics over the CORRECT group -------
    classifications = [_classify(r) for r in records]
    correct_scores = [
        float((r.get("child_metrics") or {}).get("combined_score", 0.0))
        for r, c in zip(records, classifications)
        if c == CORRECT
    ]
    if len(correct_scores) < 2:
        # Defensive: need ≥2 samples to compute meaningful std.  Fall
        # back to a deterministic mapping that still gives CORRECT a
        # positive advantage.
        mean_c = correct_scores[0] if correct_scores else 0.0
        std_c = 1.0
        print(
            f"[build_adv] WARNING: only {len(correct_scores)} correct "
            f"records — using std=1.0 fallback."
        )
    else:
        mean_c = statistics.mean(correct_scores)
        std_c = statistics.stdev(correct_scores) or 1e-6

    # Compute provisional z-scores for the correct group to get min.
    correct_zs: list[float] = []
    for r, c in zip(records, classifications):
        if c == CORRECT:
            s = float((r.get("child_metrics") or {}).get("combined_score", 0.0))
            correct_zs.append((s - mean_c) / std_c)
    min_correct_z = min(correct_zs) if correct_zs else 0.0

    wrong_penalty = min_correct_z - args.wrong_values_gap
    crash_penalty = min_correct_z - args.crashed_gap
    assert crash_penalty < wrong_penalty < min_correct_z, (
        f"Penalty ordering violated: crash={crash_penalty}, "
        f"wrong={wrong_penalty}, min_correct={min_correct_z}"
    )

    print(
        f"[build_adv] N={len(records)}  correct={classifications.count(CORRECT)}  "
        f"wrong_values={classifications.count(WRONG_VALUES)}  "
        f"crashed={classifications.count(CRASHED)}"
    )
    print(
        f"[build_adv] correct z-score stats: mean_score={mean_c:.4f} "
        f"std={std_c:.4f} min_z={min_correct_z:.3f} max_z={max(correct_zs):.3f}"
    )
    print(
        f"[build_adv] penalties: WRONG_VALUES={wrong_penalty:.3f}  "
        f"CRASHED={crash_penalty:.3f}  "
        f"(clip min={clip_min}, max={'none' if clip_max is None else clip_max})"
    )

    # --- Pass 2: build records -------------------------------------------
    # Resolve base_system / base_user from canonical prompts by default when
    # possible; fall back to synth context only when requested/needed.
    ctx_files = sorted(synth_dir.glob("iter_*_context.json"))
    first_ctx = json.loads(ctx_files[0].read_text()) if ctx_files else None
    base_system = None
    base_user = None

    canonical_ok = False
    canonical_system = ""
    canonical_user = ""
    try:
        canonical_system, canonical_user, _ = build_algotune_prompts(args.problem_id)
        canonical_ok = True
    except Exception:
        canonical_ok = False

    if args.prompt_source == "canonical":
        if not canonical_ok:
            raise ValueError(
                f"prompt_source=canonical but no canonical prompt resolver for "
                f"problem_id={args.problem_id!r}"
            )
        base_system, base_user = canonical_system, canonical_user
    elif args.prompt_source == "synth_context":
        if first_ctx is None:
            raise FileNotFoundError(
                f"prompt_source=synth_context but no iter_*_context.json in {synth_dir}"
            )
        base_system = first_ctx["base_system"]
        base_user = first_ctx["base_user"]
    else:  # auto
        if canonical_ok:
            base_system, base_user = canonical_system, canonical_user
        elif first_ctx is not None:
            base_system = first_ctx["base_system"]
            base_user = first_ctx["base_user"]
        else:
            raise FileNotFoundError(
                "No canonical prompt available and no iter_*_context.json found."
            )

    if first_ctx is not None and canonical_ok and base_system == canonical_system:
        mismatch = (
            (first_ctx.get("base_system", "") != canonical_system)
            or (first_ctx.get("base_user", "") != canonical_user)
        )
        if mismatch:
            msg = (
                "Synth context prompts differ from current canonical prompts; "
                "using canonical prompts."
            )
            if args.strict_prompt_match:
                raise ValueError(msg)
            print(f"[build_adv] WARNING: {msg}")

    out_records: list[dict] = []
    missing_synth: list[int] = []
    empty_synth: list[int] = []
    skipped_no_code: list[int] = []

    for r, cls in zip(records, classifications):
        iteration = r.get("iteration")
        child_code = r.get("child_code") or ""
        if not child_code:
            skipped_no_code.append(iteration)
            continue

        combined = float((r.get("child_metrics") or {}).get("combined_score", 0.0))
        if cls == CORRECT:
            raw_adv = (combined - mean_c) / std_c
        elif cls == WRONG_VALUES:
            raw_adv = wrong_penalty
        else:
            raw_adv = crash_penalty
        adv = max(clip_min, raw_adv)
        if clip_max is not None:
            adv = min(clip_max, adv)

        content_path = synth_dir / f"iter_{iteration:03d}_content.txt"
        if not content_path.exists():
            missing_synth.append(iteration)
            continue
        raw_reasoning = content_path.read_text()
        if not raw_reasoning.strip():
            empty_synth.append(iteration)
            continue
        cleaned = clean_synth_trace(raw_reasoning, handoff_suffix=args.handoff_suffix)
        clean_code = _strip_oe_scaffold(child_code)

        out_records.append(
            {
                "problem_id": args.problem_id,
                "phase": "phase2",
                # The trainer reads ``sample_weight``; we put the
                # clipped advantage there so the existing loss path
                # picks it up automatically.
                "sample_weight": float(adv),
                "combined_score": combined,
                "classification": cls,
                "messages": [
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": base_user},
                    {
                        "role": "assistant",
                        "thinking": cleaned,
                        "content": clean_code,
                    },
                ],
                "_meta": {
                    "iteration": iteration,
                    "raw_advantage": raw_adv,
                    "clipped_advantage": adv,
                    "reasoning_chars": len(cleaned),
                    "code_chars": len(clean_code),
                },
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    # --- Report ----------------------------------------------------------
    by_cls = {CORRECT: [], WRONG_VALUES: [], CRASHED: []}
    for rec in out_records:
        by_cls[rec["classification"]].append(rec["_meta"]["clipped_advantage"])
    print(f"\n[build_adv] wrote {len(out_records)} records → {out_path}")
    for cls, advs in by_cls.items():
        if advs:
            print(
                f"  {cls:<13} n={len(advs):>2}  "
                f"adv range=[{min(advs):+.3f}, {max(advs):+.3f}]  "
                f"mean={sum(advs) / len(advs):+.3f}"
            )
    if missing_synth:
        print(f"  missing synth output for {len(missing_synth)} iters: {missing_synth[:10]}{'...' if len(missing_synth) > 10 else ''}")
    if empty_synth:
        print(f"  empty synth output for {len(empty_synth)} iters: {empty_synth[:10]}{'...' if len(empty_synth) > 10 else ''}")
    if skipped_no_code:
        print(f"  skipped {len(skipped_no_code)} iters with empty child_code")

    # Sanity: ordering constraint holds after clipping.
    min_correct_adv = min((a for a in by_cls[CORRECT]), default=None)
    max_wrong_adv = max((a for a in by_cls[WRONG_VALUES]), default=None)
    max_crash_adv = max((a for a in by_cls[CRASHED]), default=None)
    for label, val in (("WRONG_VALUES", max_wrong_adv), ("CRASHED", max_crash_adv)):
        if min_correct_adv is not None and val is not None:
            assert val < min_correct_adv, (
                f"Clipping violated ordering: min_correct_adv={min_correct_adv} "
                f"not > {label}={val}"
            )
    print("\n[build_adv] ordering check PASSED (min_correct_adv > max_bad_adv).")


if __name__ == "__main__":
    main()

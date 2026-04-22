"""Convert OpenEvolve ``evolution_trace.jsonl`` file(s) into a DPO preference
dataset for ATLAS training.

Pairing strategy (the result is hybrid: we take *both* kinds of pairs)
----------------------------------------------------------------------
1. **Same-parent pairs** (primary / cleanest signal). Two child programs
   that share the same ``parent_id`` were sampled from essentially the
   same OpenEvolve prompt context (same parent code shown, same island,
   only minor differences in top-K siblings). Pair (chosen, rejected)
   where ``combined_score`` differs by at least ``--min-margin``. This
   gives DPO a relatively clean "same x, different y" signal.

2. **Cross-parent score-gap pairs** (augmentation). We also emit pairs
   across different parents when the score gap is *large* (``>=
   --cross-min-margin``, default 0.5). This captures the
   "runnable > garbage" and "correct > runs-but-wrong" jumps that are
   informative for teaching the base model to at least produce
   compilable Triton, even though the prompt context differs slightly.

Both kinds of pairs use the **chosen** iteration's prompt as the shared
"x" in the DPO (prompt, chosen, rejected) triple. This is a standard
approximation when chosen and rejected don't share an identical prompt.

Both pairs additionally get a ``margin`` field (``chosen_score -
rejected_score``) so downstream training can optionally weight DPO loss
by margin — the user explicitly asked that we *use* the score info
rather than reducing it to binary chosen/rejected.

Kernel kept in the dataset
--------------------------
Per the user's directive: **incorrect kernels are kept**. A pair where
the "chosen" is correct (0.5+) and "rejected" is a compile error (0.05)
teaches the base model "be correct"; a pair where both are correct but
one is faster teaches "be fast"; a pair where "chosen" runs but is
wrong (0.15) and rejected is a bare crash (0.05) teaches "at least
produce runnable Triton". The full combined_score ladder is therefore
fully exercised.

Note on the reasoning trace
---------------------------
Each trajectory's ``llm_response`` is already the **post-reasoning** text
emitted by vLLM's ``choices[0].message.content`` — that is, just the
SEARCH/REPLACE diff emitted into the Harmony ``final`` channel. The
``analysis`` channel reasoning trace is *not* present in the current
traces (vLLM's OpenAI Chat Completions endpoint doesn't surface
``reasoning_content`` separately for gpt-oss by default). To train DPO
on the full reasoning trace we'd need to either (a) reconfigure vLLM to
include reasoning in the content, or (b) parse raw Harmony tokens via
``/v1/completions``. For now we train on final-channel-only responses
and flag this as a follow-up.

Usage
-----
    python experiment/build_dpo_dataset.py \\
        --traces runs/softmax_v2_partialcredit/oe/evolution_trace.jsonl \\
                 runs/softmax_v3_twophase/oe/evolution_trace.jsonl \\
        --out data/dpo/softmax_dpo.jsonl \\
        --min-margin 0.05 --cross-min-margin 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _iter_trace(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from one evolution_trace.jsonl file."""
    with path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, path, exc)


def _record_is_usable(rec: dict[str, Any]) -> bool:
    """A record needs a non-empty LLM response, a numeric score, and the
    LLM's diff must have actually modified the parent code.

    The last filter drops "free win" records where the LLM emitted a diff
    that matched no text in the parent (so OpenEvolve kept the parent
    code verbatim and evaluated it). Those inherit the parent's score
    without the LLM contributing a real kernel — using them as DPO
    "chosen" would teach the model to emit trivially-wrong diffs just to
    inherit a good score.
    """
    resp = rec.get("llm_response") or ""
    if not isinstance(resp, str) or not resp.strip():
        return False
    metrics = rec.get("child_metrics") or {}
    if "combined_score" not in metrics:
        return False
    prompt = rec.get("prompt") or {}
    if not prompt.get("user"):
        return False
    parent_code = rec.get("parent_code", "")
    child_code = rec.get("child_code", "")
    if parent_code and child_code and parent_code == child_code:
        return False
    return True


def _build_messages(rec: dict[str, Any]) -> list[dict[str, str]]:
    """Render a trace record as the chat conversation actually sent to the
    LLM plus its response, suitable for direct chat-template tokenization.
    """
    prompt = rec.get("prompt") or {}
    return [
        {"role": "system", "content": prompt.get("system", "")},
        {"role": "user", "content": prompt.get("user", "")},
        {"role": "assistant", "content": rec.get("llm_response", "")},
    ]


def _emit_pair(
    chosen_rec: dict[str, Any],
    rejected_rec: dict[str, Any],
    pairing_type: str,
) -> dict[str, Any]:
    """Serialize one DPO triple with margin + provenance metadata."""
    cm = chosen_rec["child_metrics"]
    rm = rejected_rec["child_metrics"]
    margin = float(cm["combined_score"]) - float(rm["combined_score"])
    return {
        "chosen": _build_messages(chosen_rec),
        "rejected": _build_messages(rejected_rec),
        "chosen_score": float(cm["combined_score"]),
        "rejected_score": float(rm["combined_score"]),
        "chosen_correct": float(cm.get("correctness", 0.0)) == 1.0,
        "rejected_correct": float(rm.get("correctness", 0.0)) == 1.0,
        "margin": margin,
        "pairing_type": pairing_type,
        "chosen_iter": chosen_rec.get("iteration"),
        "rejected_iter": rejected_rec.get("iteration"),
        "parent_id_chosen": chosen_rec.get("parent_id"),
        "parent_id_rejected": rejected_rec.get("parent_id"),
    }


def build_pairs(
    records: list[dict[str, Any]],
    min_margin: float = 0.05,
    cross_min_margin: float = 0.5,
    max_cross_per_chosen: int = 8,
) -> list[dict[str, Any]]:
    """Build a list of DPO preference pairs.

    Returns a list of dicts with keys: chosen, rejected, margin, etc.
    Same-parent pairs first, then cross-parent large-gap pairs.
    """
    usable = [r for r in records if _record_is_usable(r)]
    logger.info("Trace records total=%d usable=%d", len(records), len(usable))

    pairs: list[dict[str, Any]] = []

    # 1) Same-parent pairs --------------------------------------------------
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in usable:
        pid = r.get("parent_id")
        if pid:
            by_parent[pid].append(r)

    same_parent_count = 0
    for pid, sibs in by_parent.items():
        if len(sibs) < 2:
            continue
        # Sort by score descending so iterating i < j yields chosen=i,rejected=j
        sibs_sorted = sorted(
            sibs,
            key=lambda x: x["child_metrics"]["combined_score"],
            reverse=True,
        )
        for i in range(len(sibs_sorted)):
            for j in range(i + 1, len(sibs_sorted)):
                ci = sibs_sorted[i]["child_metrics"]["combined_score"]
                cj = sibs_sorted[j]["child_metrics"]["combined_score"]
                if ci - cj >= min_margin:
                    pairs.append(_emit_pair(sibs_sorted[i], sibs_sorted[j], "same_parent"))
                    same_parent_count += 1

    # 2) Cross-parent large-gap pairs --------------------------------------
    # These are pairs across different parents; we only keep pairs where the
    # margin is large enough that it's teaching the model a qualitatively
    # different outcome tier (e.g. "runs vs crashes", "correct vs not
    # correct"). The cross-parent prompts aren't identical so we set a
    # stricter margin to avoid feeding DPO noisy almost-equivalent pairs.
    cross_count = 0
    usable_sorted = sorted(
        usable,
        key=lambda x: x["child_metrics"]["combined_score"],
        reverse=True,
    )
    # O(N^2) — this is fine for N<1000; our runs have N in the tens.
    # Note on iteration: with usable_sorted in descending score order, the
    # margin ci - cj actually *grows* as j increases (cj gets smaller), so
    # we can NOT early-break on a small margin — all further j's have a
    # LARGER margin, not a smaller one. Just walk the full N^2.
    #
    # To avoid flooding the dataset with hundreds of near-identical "best
    # correct vs each of the crashes" duplicates, we cap cross-parent pairs
    # per chosen program at ``max_cross_per_chosen``, preferring the pairs
    # with the *smallest* qualifying margin (i.e. the "hardest negatives"
    # still above the cross-margin threshold, which give DPO the richest
    # learning signal).
    for i in range(len(usable_sorted)):
        candidates: list[tuple[float, int]] = []  # (margin, j)
        for j in range(i + 1, len(usable_sorted)):
            if usable_sorted[i].get("parent_id") == usable_sorted[j].get("parent_id"):
                continue
            ci = usable_sorted[i]["child_metrics"]["combined_score"]
            cj = usable_sorted[j]["child_metrics"]["combined_score"]
            margin = ci - cj
            if margin < cross_min_margin:
                continue
            candidates.append((margin, j))
        # Take the smallest qualifying margins — these are hardest negatives.
        candidates.sort(key=lambda t: t[0])
        for _, j in candidates[:max_cross_per_chosen]:
            pairs.append(
                _emit_pair(usable_sorted[i], usable_sorted[j], "cross_parent_large_gap")
            )
            cross_count += 1

    logger.info(
        "Built %d pairs total: same_parent=%d, cross_parent_large_gap=%d",
        len(pairs),
        same_parent_count,
        cross_count,
    )
    return pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="One or more evolution_trace.jsonl files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path for the DPO dataset.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help="Minimum combined_score margin for same-parent pairs (default 0.05).",
    )
    parser.add_argument(
        "--cross-min-margin",
        type=float,
        default=0.5,
        help="Minimum combined_score margin for cross-parent pairs (default 0.5).",
    )
    parser.add_argument(
        "--max-cross-per-chosen",
        type=int,
        default=8,
        help=(
            "Cap on cross-parent pairs emitted per chosen program. Prefers "
            "hardest negatives (smallest qualifying margin). Default 8."
        ),
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.traces:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        for rec in _iter_trace(p):
            rec["_source"] = str(p)
            records.append(rec)

    pairs = build_pairs(
        records,
        min_margin=args.min_margin,
        cross_min_margin=args.cross_min_margin,
        max_cross_per_chosen=args.max_cross_per_chosen,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Summary stats for the user.
    same_parent = sum(1 for p in pairs if p["pairing_type"] == "same_parent")
    cross = sum(1 for p in pairs if p["pairing_type"] == "cross_parent_large_gap")
    correctness_flips = sum(
        1 for p in pairs if p["chosen_correct"] and not p["rejected_correct"]
    )
    both_correct = sum(1 for p in pairs if p["chosen_correct"] and p["rejected_correct"])
    both_broken = sum(
        1
        for p in pairs
        if not p["chosen_correct"] and not p["rejected_correct"]
    )
    logger.info("Wrote %d pairs to %s", len(pairs), out_path)
    logger.info(
        "  same_parent=%d  cross_parent=%d", same_parent, cross,
    )
    logger.info(
        "  correct-vs-incorrect=%d  correct-vs-correct=%d  incorrect-vs-incorrect=%d",
        correctness_flips,
        both_correct,
        both_broken,
    )


if __name__ == "__main__":
    main()

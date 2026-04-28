"""Build an ATLAS-RLM memory bank from OE trajectories.

An "RLM memory bank" is a single text file containing numbered entries,
one per OE trajectory, that the base model can search over at inference
time.  Each entry distills the trajectory into:

  1. Outcome tag: ``correct`` / ``wrong_values`` / ``crashed`` + score.
  2. Condensed lesson text (the first ~N sentences of the cleaned
     synthetic reasoning trace that base gpt-oss-20b produced when asked
     to explain the trajectory — i.e. the base model has already "looked
     at" the trajectory).
  3. Code excerpt: first ``--code-lines`` lines of the final program.

The result is designed to fit comfortably in gpt-oss-20b's context window
so the inference-time "search call" can see every entry at once.

Usage::

    python experiment/build_rlm_memory_bank.py \
        --in data/synth_softmax_all_advantage_v1.jsonl \
        --out data/rlm_memory_softmax.txt \
        --max-lesson-chars 900 \
        --code-lines 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _take_sentences(text: str, max_chars: int) -> str:
    """Greedily take leading sentences until budget is hit."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    out: list[str] = []
    remaining = max_chars
    # Split on hard stops so we don't chop mid-clause.
    for sent in text.replace("\n\n", ". ").split(". "):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) + 2 > remaining:
            break
        out.append(sent)
        remaining -= len(sent) + 2
    return ". ".join(out).strip() + "."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-lesson-chars", type=int, default=900)
    p.add_argument("--code-lines", type=int, default=40)
    args = p.parse_args()

    recs = [json.loads(l) for l in Path(args.in_path).read_text().splitlines() if l.strip()]
    # Stable ordering: highest score first (most useful lessons up top).
    recs.sort(key=lambda r: (-float(r.get("combined_score", 0.0)), r["_meta"]["iteration"]))

    parts: list[str] = []
    parts.append(
        "# ATLAS-RLM memory bank (softmax OE run)\n"
        "# Each entry below summarises one attempt by the base model at the\n"
        "# softmax Triton-kernel task.  Entries are sorted by score (best first).\n"
        f"# Total entries: {len(recs)}\n"
    )

    for idx, r in enumerate(recs, start=1):
        cls = r["classification"]
        score = float(r.get("combined_score", 0.0))
        iteration = r["_meta"].get("iteration", -1)
        thinking = (r["messages"][2].get("thinking") or "").strip()
        code = (r["messages"][2].get("content") or "").strip()

        lesson = _take_sentences(thinking, args.max_lesson_chars)
        code_lines = code.splitlines()[: args.code_lines]
        excerpt = "\n".join(code_lines)
        if len(code.splitlines()) > args.code_lines:
            excerpt += f"\n# ...(+{len(code.splitlines()) - args.code_lines} more lines truncated)"

        entry = (
            f"\n============================\n"
            f"ENTRY {idx}  [{cls}, score={score:.3f}, iter={iteration}]\n"
            f"============================\n"
            f"## Lesson\n{lesson or '(no reasoning captured)'}\n\n"
            f"## Code excerpt\n```python\n{excerpt}\n```\n"
        )
        parts.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))

    # Quick stats
    total_chars = sum(len(p) for p in parts)
    avg_chars = total_chars // max(1, len(recs))
    # Rough token estimate: ~4 chars/token for English+code
    est_tokens = total_chars // 4
    print(f"[rlm_bank] wrote {len(recs)} entries → {out_path}")
    print(f"[rlm_bank] total chars={total_chars}  avg chars/entry={avg_chars}")
    print(f"[rlm_bank] estimated tokens ≈ {est_tokens}")


if __name__ == "__main__":
    main()

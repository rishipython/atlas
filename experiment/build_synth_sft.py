"""Build a tiny reward-weighted SFT dataset from synthesized reasoning.

Given a ``synth_reasoning`` output directory (downloaded from the Modal
volume) and a chosen sample index, this script produces a one-record
JSONL dataset of the form::

    {
      "messages": [
        {"role": "system",   "content": base_system},
        {"role": "user",     "content": base_user},
        {"role": "assistant", "thinking": synth_reasoning, "content": final_code}
      ],
      "sample_weight": 1.0,
      "phase": "phase2",
      ...
    }

matching the schema consumed by ``train_atlas_sft.py``.

The synthesized reasoning is *post-processed* to:
  1. truncate at the first triple-backtick-fenced code block longer than
     3 lines (short scratch snippets in fences are allowed, but a full
     kernel body in a fence is not);
  2. truncate at runs of 3+ consecutive lines that share the same
     opening 4 words (the "Now we need to..." degeneration attractor);
  3. trim the final incomplete sentence;
  4. append a clean handoff line.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_LEADING_REPEAT_RE = re.compile(r"^(\s*[A-Za-z,]+(?:\s+[A-Za-z,]+){0,3})")


def _strip_long_fenced_blocks(text: str) -> str:
    """Truncate at the first ```-fenced block whose content exceeds 3 lines.

    Short inline fences (e.g. a 1-4 line sketch) are kept because they
    mirror the style of real gpt-oss-20b reasoning.  A long fenced
    block, though, is almost always the model giving up on the
    "no final code" instruction and dumping the kernel — we chop it.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            # Find the closing fence
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            block_len = j - i - 1  # inner content lines
            if block_len > 3:
                # Long code block — truncate everything from here.
                break
            # Short block; keep verbatim including the closing fence (if present).
            out.extend(lines[i : j + 1])
            i = j + 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _strip_repetition_tail(text: str) -> str:
    """Cut at the start of a run of 3+ consecutive lines with the same
    leading 3-4 words (a common late-sequence degeneration)."""
    lines = text.splitlines()
    if len(lines) < 4:
        return text
    last_prefix: str | None = None
    run = 0
    cut_at: int | None = None
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        m = _LEADING_REPEAT_RE.match(ln)
        if not m:
            last_prefix = None
            run = 0
            continue
        prefix = m.group(1).strip().lower()
        if len(prefix) < 8:  # ignore ultra-short prefixes
            last_prefix = None
            run = 0
            continue
        if prefix == last_prefix:
            run += 1
            if run >= 2 and cut_at is None:
                cut_at = i - 2  # start of the repeating run
        else:
            last_prefix = prefix
            run = 1
    if cut_at is not None:
        return "\n".join(lines[:cut_at])
    return text


def _trim_incomplete_tail(text: str) -> str:
    """Drop the final paragraph if it ends without a terminal punctuation.

    Avoids training on half-truncated sentences caused by finish=length.
    """
    text = text.rstrip()
    if not text:
        return text
    # If the text already ends in terminal punctuation, keep as is.
    if text[-1] in ".!?`)":
        return text
    # Otherwise, drop back to the last occurrence of ". " or "?\n" etc.
    for terminator in (". ", ".\n", "?\n", "!\n"):
        idx = text.rfind(terminator)
        if idx != -1:
            return text[: idx + 1].rstrip()
    return text


def clean_synth_trace(raw: str) -> str:
    """Apply the full post-processing chain."""
    cleaned = _strip_long_fenced_blocks(raw)
    cleaned = _strip_repetition_tail(cleaned)
    cleaned = _trim_incomplete_tail(cleaned)
    cleaned = cleaned.rstrip()
    if not cleaned.endswith(("kernel now.", "kernel.", "write it.")):
        cleaned += "\n\nOK, I think I have the design. Time to write the kernel."
    return cleaned


def _strip_oe_scaffold(code: str) -> str:
    """Drop OpenEvolve EVOLVE-BLOCK markers + the starter docstring.

    The child_code stored in evolution_trace.jsonl contains the scaffold
    comments from the starter program.  We want pure runnable Python.
    """
    lines = code.splitlines()
    keep: list[str] = []
    in_block = False
    started = False
    for ln in lines:
        if "EVOLVE-BLOCK-START" in ln:
            in_block = True
            continue
        if "EVOLVE-BLOCK-END" in ln:
            in_block = False
            continue
        if in_block or started:
            keep.append(ln)
            started = True
    stripped = "\n".join(keep).strip()
    return stripped or code  # fall back if markers were absent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synth-dir",
        required=True,
        help="local path to the downloaded synth output (e.g. "
        "/tmp/synth_v3_artifacts/synth_softmax_v3)",
    )
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--sample-weight",
        type=float,
        default=1.0,
        help="reward weight for this record (single-record dataset so "
        "this is effectively just the loss scale).",
    )
    args = parser.parse_args()

    synth_dir = Path(args.synth_dir)
    ctx = json.loads((synth_dir / "target_context.json").read_text())
    content = (
        synth_dir / f"sample_{args.sample_idx:02d}_content.txt"
    ).read_text()

    raw_len = len(content)
    cleaned_reasoning = clean_synth_trace(content)
    clean_len = len(cleaned_reasoning)
    final_code = _strip_oe_scaffold(ctx["final_code"])

    record = {
        "problem_id": ctx["problem_id"],
        "phase": "phase2",
        "sample_weight": float(args.sample_weight),
        "combined_score": ctx["final_score"],
        "messages": [
            {"role": "system", "content": ctx["base_system"]},
            {"role": "user", "content": ctx["base_user"]},
            {
                "role": "assistant",
                "thinking": cleaned_reasoning,
                "content": final_code,
            },
        ],
        "_meta": {
            "synth_source": str(synth_dir),
            "sample_idx": args.sample_idx,
            "raw_content_chars": raw_len,
            "cleaned_reasoning_chars": clean_len,
            "final_code_chars": len(final_code),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"[build_synth_sft] wrote 1 record to {out}\n"
        f"  problem_id      = {ctx['problem_id']}\n"
        f"  raw reasoning   = {raw_len} chars\n"
        f"  cleaned reason  = {clean_len} chars\n"
        f"  final code      = {len(final_code)} chars\n"
        f"  reasoning head:\n    " + cleaned_reasoning[:300].replace("\n", "\n    ")
    )
    print(
        "\n  reasoning tail:\n    "
        + cleaned_reasoning[-400:].replace("\n", "\n    ")
    )


if __name__ == "__main__":
    main()

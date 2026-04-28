"""Build FFT-convolution SFT datasets from OE trace + synthesized reasoning.

Outputs three datasets:
1) Phase-1 correctness dataset:
   - all trajectories
   - sample_weight = +1 for correctness>=threshold, else -1
2) Phase-2 speedup dataset:
   - correct-only trajectories
   - sample_weight = z-score(speedup) within correct-only set
3) Best-only dataset:
   - single correct trajectory with max speedup
   - sample_weight = 1.0

This script is intentionally task-agnostic in implementation, but it is
primarily used for fft_convolution runs where the user wants OE "best"
to mean "max speedup among correct trajectories", not best combined_score.
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


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_base_prompts(
    synth_dir: Path,
    problem_id: str,
    prompt_source: str,
    strict_prompt_match: bool,
) -> tuple[str, str]:
    canonical_system, canonical_user, _ = build_algotune_prompts(problem_id)
    ctx_files = sorted(synth_dir.glob("iter_*_context.json"))
    if prompt_source == "canonical":
        if ctx_files:
            first_ctx = json.loads(ctx_files[0].read_text())
            mismatch = (
                (first_ctx.get("base_system", "") != canonical_system)
                or (first_ctx.get("base_user", "") != canonical_user)
            )
            if mismatch:
                msg = (
                    "Synth context prompts differ from current canonical prompts; "
                    "using canonical prompts anyway."
                )
                if strict_prompt_match:
                    raise ValueError(msg)
                print(f"[build_fft] WARNING: {msg}")
        return canonical_system, canonical_user

    if prompt_source == "synth_context":
        if not ctx_files:
            raise FileNotFoundError(
                f"prompt_source=synth_context but no iter_*_context.json in {synth_dir}"
            )
        first_ctx = json.loads(ctx_files[0].read_text())
        return first_ctx["base_system"], first_ctx["base_user"]

    raise ValueError(f"Unknown prompt_source={prompt_source!r}")


def _load_synth_reasoning_map(synth_dir: Path) -> dict[int, str]:
    """Return iteration -> synthesized reasoning text.

    Preferred source is iter_NNN_content.txt files. If those are not present,
    fallback to all_samples.json (task_id is iter_NNN).
    """
    out: dict[int, str] = {}
    for p in sorted(synth_dir.glob("iter_*_content.txt")):
        stem = p.stem  # iter_001_content
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            iteration = int(parts[1])
        except ValueError:
            continue
        out[iteration] = p.read_text()
    if out:
        return out

    all_samples = synth_dir / "all_samples.json"
    if all_samples.exists():
        arr = json.loads(all_samples.read_text())
        for rec in arr:
            task_id = str(rec.get("task_id", ""))
            if not task_id.startswith("iter_"):
                continue
            try:
                iteration = int(task_id.split("_", 1)[1])
            except ValueError:
                continue
            out[iteration] = rec.get("content") or ""
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="OE evolution_trace.jsonl")
    p.add_argument("--synth-dir", required=True, help="dir with iter_*_{context,content}.txt")
    p.add_argument("--problem-id", required=True)
    p.add_argument("--out-p1", required=True)
    p.add_argument("--out-p2", required=True)
    p.add_argument("--out-best", required=True)
    p.add_argument("--handoff-suffix", default="it")
    p.add_argument("--correct-threshold", type=float, default=0.99)
    p.add_argument(
        "--prompt-source",
        choices=["canonical", "synth_context"],
        default="canonical",
        help=(
            "Where to source system/user prompts. "
            "Default canonical ensures train/eval prompt alignment."
        ),
    )
    p.add_argument(
        "--strict-prompt-match",
        action="store_true",
        help=(
            "When prompt-source=canonical and synth context exists, fail if "
            "stored synth prompts differ from canonical prompts."
        ),
    )
    args = p.parse_args()

    trace_path = Path(args.trace)
    synth_dir = Path(args.synth_dir)
    out_p1 = Path(args.out_p1)
    out_p2 = Path(args.out_p2)
    out_best = Path(args.out_best)

    records = _load_jsonl(trace_path)
    base_system, base_user = _read_base_prompts(
        synth_dir=synth_dir,
        problem_id=args.problem_id,
        prompt_source=args.prompt_source,
        strict_prompt_match=bool(args.strict_prompt_match),
    )
    synth_reasoning_map = _load_synth_reasoning_map(synth_dir)

    assembled: list[dict] = []
    missing_synth: list[int] = []
    empty_synth: list[int] = []
    skipped_no_code: list[int] = []

    for r in records:
        iteration = int(r.get("iteration"))
        child_code = r.get("child_code") or ""
        if not child_code:
            skipped_no_code.append(iteration)
            continue

        metrics = r.get("child_metrics") or {}
        correctness = float(metrics.get("correctness", 0.0) or 0.0)
        speedup = float(metrics.get("speedup", 0.0) or 0.0)
        is_correct = correctness >= args.correct_threshold

        raw_reasoning = synth_reasoning_map.get(iteration)
        if raw_reasoning is None:
            missing_synth.append(iteration)
            continue
        if not raw_reasoning.strip():
            empty_synth.append(iteration)
            continue

        cleaned_reasoning = clean_synth_trace(
            raw_reasoning, handoff_suffix=args.handoff_suffix
        )
        clean_code = _strip_oe_scaffold(child_code)
        messages = [
            {"role": "system", "content": base_system},
            {"role": "user", "content": base_user},
            {
                "role": "assistant",
                "thinking": cleaned_reasoning,
                "content": clean_code,
            },
        ]
        assembled.append(
            {
                "iteration": iteration,
                "correctness": correctness,
                "speedup": speedup,
                "is_correct": is_correct,
                "messages": messages,
                "reasoning_chars": len(cleaned_reasoning),
                "code_chars": len(clean_code),
            }
        )

    # Phase 1: correctness (+1/-1) over all assembled records.
    p1_records: list[dict] = []
    for a in assembled:
        weight = 1.0 if a["is_correct"] else -1.0
        p1_records.append(
            {
                "problem_id": args.problem_id,
                "phase": "phase1_correctness",
                "sample_weight": weight,
                "messages": a["messages"],
                "_meta": {
                    "iteration": a["iteration"],
                    "correctness": a["correctness"],
                    "speedup": a["speedup"],
                    "is_correct": a["is_correct"],
                    "reasoning_chars": a["reasoning_chars"],
                    "code_chars": a["code_chars"],
                },
            }
        )

    # Phase 2: correct-only, speedup z-score.
    correct_only = [a for a in assembled if a["is_correct"]]
    speedups = [a["speedup"] for a in correct_only]
    if speedups:
        mean_s = statistics.mean(speedups)
        std_s = statistics.stdev(speedups) if len(speedups) > 1 else 1.0
        std_s = std_s or 1e-6
    else:
        mean_s, std_s = 0.0, 1.0

    p2_records: list[dict] = []
    for a in correct_only:
        z = (a["speedup"] - mean_s) / std_s
        p2_records.append(
            {
                "problem_id": args.problem_id,
                "phase": "phase2_speedup",
                "sample_weight": float(z),
                "messages": a["messages"],
                "_meta": {
                    "iteration": a["iteration"],
                    "correctness": a["correctness"],
                    "speedup": a["speedup"],
                    "z_speedup": float(z),
                    "reasoning_chars": a["reasoning_chars"],
                    "code_chars": a["code_chars"],
                },
            }
        )

    # Best-only: max speedup among correct trajectories.
    best_record: dict | None = None
    if correct_only:
        best_record = max(correct_only, key=lambda a: a["speedup"])
    best_out_records: list[dict] = []
    if best_record is not None:
        best_out_records.append(
            {
                "problem_id": args.problem_id,
                "phase": "best_speedup_only",
                "sample_weight": 1.0,
                "messages": best_record["messages"],
                "_meta": {
                    "iteration": best_record["iteration"],
                    "correctness": best_record["correctness"],
                    "speedup": best_record["speedup"],
                    "reasoning_chars": best_record["reasoning_chars"],
                    "code_chars": best_record["code_chars"],
                },
            }
        )

    for out in (out_p1, out_p2, out_best):
        out.parent.mkdir(parents=True, exist_ok=True)

    with out_p1.open("w") as f:
        for rec in p1_records:
            f.write(json.dumps(rec) + "\n")
    with out_p2.open("w") as f:
        for rec in p2_records:
            f.write(json.dumps(rec) + "\n")
    with out_best.open("w") as f:
        for rec in best_out_records:
            f.write(json.dumps(rec) + "\n")

    print(
        f"[build_fft] source={len(records)} assembled={len(assembled)} "
        f"correct={len(correct_only)}"
    )
    print(f"[build_fft] wrote phase1: {len(p1_records)} -> {out_p1}")
    print(f"[build_fft] wrote phase2: {len(p2_records)} -> {out_p2}")
    print(f"[build_fft] wrote best-only: {len(best_out_records)} -> {out_best}")
    if correct_only:
        print(
            f"[build_fft] speedup(correct) mean={mean_s:.6f} std={std_s:.6f} "
            f"best={best_record['speedup']:.6f} iter={best_record['iteration']}"
        )
    if missing_synth:
        print(
            f"[build_fft] missing synth content for {len(missing_synth)} iters: "
            f"{missing_synth[:10]}{'...' if len(missing_synth) > 10 else ''}"
        )
    if empty_synth:
        print(
            f"[build_fft] empty synth content for {len(empty_synth)} iters: "
            f"{empty_synth[:10]}{'...' if len(empty_synth) > 10 else ''}"
        )
    if skipped_no_code:
        print(
            f"[build_fft] skipped no-child-code for {len(skipped_no_code)} iters: "
            f"{skipped_no_code[:10]}{'...' if len(skipped_no_code) > 10 else ''}"
        )


if __name__ == "__main__":
    main()

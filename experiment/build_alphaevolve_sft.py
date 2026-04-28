"""Build a reward-weighted SFT dataset from an AlphaEvolve OE trajectory.

Sibling of ``build_sft_dataset.py`` for AlphaEvolve math problems.  The
schema of the output JSONL files matches the kernel builder so the
existing ``train_atlas_sft.py`` trainer can consume them unchanged.

What's different from the kernel builder
----------------------------------------
* No Triton requirement — AE programs are plain numpy/scipy.
* No KERNEL_PROBLEMS lookup — base prompts are reconstructed from the
  upstream files snapshotted into the run dir
  (``initial_program.py`` + ``config.yaml`` ``prompt.system_message``).
* ``min_score`` is a fraction of the AlphaEvolve benchmark (the
  upstream evaluators normalize ``combined_score`` so 1.0 = matching the
  state-of-the-art reported by AlphaEvolve).  Default 0.6 keeps records
  that achieve at least 60% of SOTA — sane floor for distillation,
  rejects the majority of crashes/no-ops.

Output schema (per record, matches the kernel SFT JSONL exactly)::

    {
      "problem_id": "circle_packing_rect",
      "phase":      "phase1" | "phase2",
      "source_iter": 12,
      "source_file": "runs/ae_oe_base_p1/oe/evolution_trace.jsonl",
      "messages": [
        {"role": "system",   "content": ...},
        {"role": "user",     "content": ...},
        {"role": "assistant","content": <child code>, "thinking": ...?}
      ],
      "raw_score":     0.74,
      "sample_weight": 0.74,        # or rank-normalized in [0,1]
      "combined_score": 0.74,
      "_meta": { ... }
    }

Usage
-----
    python experiment/build_alphaevolve_sft.py \\
        --run-dir runs/ae_oe_base_p1 \\
        --problem-id circle_packing_rect \\
        --phase both \\
        --out-phase1 data/sft/ae_p1_phase1.jsonl \\
        --out-phase2 data/sft/ae_p1_phase2.jsonl \\
        --min-score 0.6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------
def _iter_trace(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skipping malformed line %d in %s: %s", line_num, path, exc
                )


# OpenEvolve injects ``# EVOLVE-BLOCK-START`` / ``# EVOLVE-BLOCK-END``
# marker comments into ``child_code``; strip them so the training target
# is a clean standalone program (these markers would never appear in a
# response generated from the base prompt at inference time).
_EVOLVE_MARKER_RE = re.compile(
    r"^\s*#\s*EVOLVE-BLOCK-(START|END)\s*$", re.MULTILINE
)


def _clean_child_code(code: str) -> str:
    return _EVOLVE_MARKER_RE.sub("", code).strip() + "\n"


# ---------------------------------------------------------------------------
# Per-record filter
# ---------------------------------------------------------------------------
def _record_is_usable(rec: dict[str, Any], *, min_score: float) -> bool:
    resp = rec.get("llm_response") or ""
    if not isinstance(resp, str) or not resp.strip():
        return False
    cm = rec.get("child_metrics") or {}
    if "combined_score" not in cm:
        return False
    if float(cm["combined_score"]) < min_score:
        return False
    prompt = rec.get("prompt") or {}
    if not prompt.get("user") or not prompt.get("system"):
        return False
    parent_code = rec.get("parent_code", "")
    child_code = rec.get("child_code", "")
    if not child_code:
        return False
    # Drop "free wins" where the LLM's diff matched nothing and OE just
    # re-evaluated the parent with no new information.
    if parent_code and parent_code == child_code:
        return False
    return True


# ---------------------------------------------------------------------------
# Base-prompt construction (Phase 2)
# ---------------------------------------------------------------------------
def _build_base_prompt(
    system_message: str, initial_program: str, problem_id: str
) -> tuple[str, str]:
    """Return (system, user) for the base / inference-time prompt.

    The system message is the per-problem prompt copied verbatim from
    the upstream ``config.yaml``.  The user message asks for a full
    solution starting from the upstream ``initial_program.py`` —
    matching what ``eval_alphaevolve.py`` will send at evaluation time.
    """
    user = (
        f"Below is the starter program for the `{problem_id}` problem.\n\n"
        f"```python\n{initial_program.rstrip()}\n```\n\n"
        f"Your task: write a complete, runnable Python file that improves "
        f"on this starter to MAXIMIZE the `combined_score` reported by "
        f"the problem's evaluator (a `combined_score` ≥ 1.0 means matching "
        f"or beating the AlphaEvolve state-of-the-art).  Keep the same "
        f"function signature(s) the evaluator expects; preserve safe "
        f"`if __name__ == \"__main__\":` guards so importing the file is "
        f"side-effect-free.  Return a single ```python fenced block "
        f"containing the full solution."
    )
    return system_message, user


# ---------------------------------------------------------------------------
# Reasoning-trace join (matches the kernel builder)
# ---------------------------------------------------------------------------
def _prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_reasoning_index(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        logger.warning(
            "reasoning-trace file not found at %s; proceeding without", path
        )
        return out
    kept = 0
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = rec.get("content")
            reasoning = rec.get("reasoning_content")
            h = rec.get("prompt_hash")
            if not h or not content or not reasoning:
                continue
            out[h] = reasoning
            kept += 1
    logger.info("loaded reasoning traces: %d entries from %s", kept, path)
    return out


# ---------------------------------------------------------------------------
# Rank normalize (per-problem, robust to scale)
# ---------------------------------------------------------------------------
def _rank_normalize(scores: list[float]) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    for new_rank, i in enumerate(order):
        ranks[i] = new_rank / max(1, n - 1)
    return ranks


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_records(
    trace_records: list[dict[str, Any]],
    *,
    problem_id: str,
    base_system: str,
    initial_program: str,
    emit_phase1: bool,
    emit_phase2: bool,
    min_score: float,
    rank_normalize: bool,
    reasoning_index: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [r for r in trace_records if _record_is_usable(r, min_score=min_score)]
    logger.info("records: total=%d usable=%d", len(trace_records), len(usable))
    if not usable:
        return [], []

    scores = [float(r["child_metrics"]["combined_score"]) for r in usable]
    weights = _rank_normalize(scores) if rank_normalize else scores
    logger.info(
        "problem=%s n=%d combined_score range=[%.3f, %.3f] weight-scheme=%s",
        problem_id,
        len(usable),
        min(scores),
        max(scores),
        "rank" if rank_normalize else "raw_score",
    )

    if emit_phase2:
        base_user_template_system, base_user = _build_base_prompt(
            base_system, initial_program, problem_id
        )
        # Keep the system identical to the OE one (which the upstream
        # config sets as ``prompt.system_message``); only the user-side
        # message differs between phase 1 and phase 2.
        base_user_system = base_user_template_system
    else:
        base_user_system = base_user = None

    phase1_records: list[dict[str, Any]] = []
    phase2_records: list[dict[str, Any]] = []

    for r, w in zip(usable, weights):
        oe_system = r["prompt"]["system"]
        oe_user = r["prompt"]["user"]
        clean_code = _clean_child_code(r["child_code"])

        oe_messages_for_hash = [
            {"role": "system", "content": oe_system},
            {"role": "user", "content": oe_user},
        ]
        reasoning = reasoning_index.get(_prompt_hash(oe_messages_for_hash))

        cm = r["child_metrics"]
        common = {
            "problem_id": problem_id,
            "source_iter": r.get("iteration"),
            "source_file": r.get("_source"),
            "raw_score": float(cm["combined_score"]),
            "sample_weight": float(w),
            "combined_score": float(cm["combined_score"]),
            # Surface every metric the AE evaluator emitted (eval_time,
            # radii_sum, c1, etc.) so downstream analysis isn't blind.
            "all_metrics": {k: v for k, v in cm.items()},
        }
        if reasoning:
            common["reasoning"] = reasoning

        assistant_msg: dict[str, str] = {
            "role": "assistant",
            "content": clean_code,
        }
        if reasoning:
            assistant_msg["thinking"] = reasoning

        if emit_phase1:
            phase1_records.append(
                {
                    **common,
                    "phase": "phase1",
                    "messages": [
                        {"role": "system", "content": oe_system},
                        {"role": "user", "content": oe_user},
                        assistant_msg,
                    ],
                }
            )
        if emit_phase2:
            phase2_records.append(
                {
                    **common,
                    "phase": "phase2",
                    "messages": [
                        {"role": "system", "content": base_user_system},
                        {"role": "user", "content": base_user},
                        assistant_msg,
                    ],
                }
            )

    return phase1_records, phase2_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a downloaded run dir (must contain config.yaml, "
        "initial_program.py, oe/evolution_trace.jsonl, and optionally "
        "oe/reasoning_trace.jsonl).",
    )
    parser.add_argument(
        "--problem-id",
        required=True,
        help="Stamped onto each record's ``problem_id`` field.",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        default="both",
    )
    parser.add_argument("--out-phase1", default=None)
    parser.add_argument("--out-phase2", default=None)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Drop records with combined_score below this (default 0.6).",
    )
    parser.add_argument(
        "--rank-normalize-weights",
        action="store_true",
        help="Per-problem rank-normalize sample_weight into [0, 1].",
    )
    args = parser.parse_args()

    emit_phase1 = args.phase in ("1", "both")
    emit_phase2 = args.phase in ("2", "both")
    if emit_phase1 and not args.out_phase1:
        parser.error("--out-phase1 is required when --phase is 1 or both")
    if emit_phase2 and not args.out_phase2:
        parser.error("--out-phase2 is required when --phase is 2 or both")

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config.yaml"
    init_path = run_dir / "initial_program.py"
    trace_path = run_dir / "oe" / "evolution_trace.jsonl"
    reasoning_path = run_dir / "oe" / "reasoning_trace.jsonl"
    for required in (cfg_path, init_path, trace_path):
        if not required.exists():
            parser.error(f"required file missing: {required}")

    import yaml  # local import — only this CLI uses it

    cfg = yaml.safe_load(cfg_path.read_text())
    base_system = cfg.get("prompt", {}).get("system_message", "")
    if not base_system:
        parser.error(
            f"config.yaml at {cfg_path} has no ``prompt.system_message`` — "
            f"cannot construct phase-2 base prompt."
        )
    initial_program = init_path.read_text()

    records: list[dict[str, Any]] = []
    for rec in _iter_trace(trace_path):
        rec["_source"] = str(trace_path)
        records.append(rec)
    logger.info("loaded %d raw trace records from %s", len(records), trace_path)

    reasoning_index = _load_reasoning_index(reasoning_path)

    phase1, phase2 = build_records(
        records,
        problem_id=args.problem_id,
        base_system=base_system,
        initial_program=initial_program,
        emit_phase1=emit_phase1,
        emit_phase2=emit_phase2,
        min_score=args.min_score,
        rank_normalize=args.rank_normalize_weights,
        reasoning_index=reasoning_index,
    )

    if emit_phase1:
        out = Path(args.out_phase1)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in phase1:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("wrote %d phase-1 records to %s", len(phase1), out)
    if emit_phase2:
        out = Path(args.out_phase2)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in phase2:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("wrote %d phase-2 records to %s", len(phase2), out)


if __name__ == "__main__":
    main()

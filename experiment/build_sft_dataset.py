"""Convert OpenEvolve ``evolution_trace.jsonl`` files into reward-weighted
SFT datasets for the two-phase ATLAS training.

Why SFT, not DPO?
-----------------
The user vetoed DPO after an end-to-end run produced a regressing ATLAS.
The root cause was a train/eval distribution mismatch: DPO trained the
model to output OpenEvolve-style SEARCH/REPLACE diffs, but at eval time
we asked for a full fenced program.  The new plan is:

* **Phase 1 (``--phase 1``)** — SFT on (OE system + OE user prompt) →
  full child program.  This learns to produce good programs from the
  dense-context prompt OpenEvolve actually sends at search time.
* **Phase 2 (``--phase 2``)** — SFT on (base system + base user prompt)
  → same full child program.  This is the "bridge" step that teaches
  the model to produce the same good programs from the short,
  user-facing base prompt (what eval/downstream use will actually
  send).  We mix in a configurable fraction of Phase-1 data as
  *replay* so the model doesn't forget the OE-context behavior.

Both phases learn **full programs**, not diffs.  That's the invariant
the user asked for.

Record filters
--------------
A record is kept iff:

1. ``llm_response`` is a non-empty string (so the LLM actually answered
   something, even if partial);
2. ``child_metrics.combined_score`` is present;
3. ``parent_code != child_code`` — this drops "free win" records where
   the diff matched nothing and OpenEvolve re-evaluated the parent
   verbatim, inheriting its score without the LLM contributing a real
   program;
4. ``child_code`` contains ``triton`` *or* ``@triton.jit`` (optional via
   ``--require-triton``) — on by default so we don't train on "the
   reference PyTorch copy is correct" records, which teach nothing;
5. ``combined_score >= --min-score`` (default 0.5; keeps the tier of
   programs that at least compiled and produced a numerically-close
   output on one of the test shapes).

Reward weighting
----------------
Each surviving record gets ``sample_weight = combined_score``
(optionally rank-normalized per problem via ``--rank-normalize-weights``,
which makes cross-problem mixing less sensitive to absolute reward
scales).  The trainer multiplies per-example NLL by this weight — a
RAFT-style reward-weighted objective.  Records that merely "compiled"
(~0.5) still contribute, but records that were correct *and* fast (~0.8)
contribute more.

Reasoning-trace joining (future-ready)
--------------------------------------
``--reasoning-trace`` optionally points at the JSONL produced by our
``sitecustomize`` side-logger (enabled when vLLM is launched with
``--reasoning-parser openai_gptoss`` — see ``openevolve_runner.py``).
When present, each record is joined by ``prompt_hash`` and gets a
``reasoning`` field containing the Harmony ``analysis`` channel.  The
current softmax run we're training on does NOT have this file, so the
``reasoning`` field will be absent.  The next OE re-run (see
``openevolve_runner.py`` post-change) will produce it automatically
and the builder will pick it up with no further changes.

Output format (one JSONL record per surviving evolution trace record,
each emitted as either ``phase1``, ``phase2``, or *both* depending on
``--phase``):

    {
      "phase": "phase1" | "phase2",
      "problem_id": "softmax",
      "source_iter": 12,
      "source_file": "runs/softmax_v3_twophase/oe/evolution_trace.jsonl",
      "messages": [
        {"role": "system",   "content": ...},
        {"role": "user",     "content": ...},
        {"role": "assistant","content": ...}
      ],
      "reasoning": "..."              // optional, only if joined
      "raw_score": 0.808,
      "sample_weight": 0.808,         // or per-problem-normalized rank
      "correct": true,
      "speedup": 0.612,
    }

Usage
-----
    python experiment/build_sft_dataset.py \\
        --traces runs/softmax_v3_twophase/oe/evolution_trace.jsonl \\
        --phase both \\
        --out-phase1 data/sft/softmax_phase1.jsonl \\
        --out-phase2 data/sft/softmax_phase2.jsonl \\
        --min-score 0.5
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
# Evolution trace I/O + filtering
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
                logger.warning("skipping malformed line %d in %s: %s", line_num, path, exc)


def _record_problem_id(rec: dict[str, Any]) -> str | None:
    """Best-effort problem_id extraction.

    OpenEvolve doesn't stamp the problem_id into each record, so we
    infer it from the ``_source`` path the caller attached (e.g.
    ``runs/softmax_v3.../evolution_trace.jsonl``).  That's fragile but
    works for our naming convention — each run dir is prefixed with
    the problem id.  Callers can override via ``--problem-id-override``.
    """
    src = rec.get("_source")
    if not src:
        return None
    # e.g. runs/softmax_v3_twophase/oe/evolution_trace.jsonl
    parts = Path(src).parts
    try:
        runs_idx = parts.index("runs")
        run_dir = parts[runs_idx + 1]
    except (ValueError, IndexError):
        return None
    # Take the leading slug before the first underscore: "softmax_v3_twophase" -> "softmax"
    return run_dir.split("_", 1)[0]


# Pattern to strip the OE-injected evolve-block scaffolding from child_code
# so the training target is a clean, standalone Triton program.  We keep
# the module docstring (it's short and often contains useful intent) but
# remove the ``# EVOLVE-BLOCK-START/END`` marker lines which are pure
# OpenEvolve machinery that the base prompt would never produce.
_EVOLVE_MARKER_RE = re.compile(r"^\s*#\s*EVOLVE-BLOCK-(START|END)\s*$", re.MULTILINE)


def _clean_child_code(code: str) -> str:
    return _EVOLVE_MARKER_RE.sub("", code).strip() + "\n"


def _record_is_usable(
    rec: dict[str, Any],
    *,
    min_score: float,
    require_triton: bool,
) -> bool:
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
    # "Free win": LLM's diff matched nothing, OE kept parent verbatim.
    # Those inherit the parent's score without teaching the model anything.
    if parent_code and parent_code == child_code:
        return False
    if require_triton:
        # "triton" appears in the module docstring regardless of the body,
        # so we check the actual body (everything after the first `import`).
        body = child_code
        if "@triton.jit" not in body and "import triton" not in body:
            return False
    return True


# ---------------------------------------------------------------------------
# Base-prompt construction (Phase 2).  Must match ``eval_standalone._build_
# standalone_prompts`` exactly — the whole point of Phase 2 is that the
# model sees at *inference* time the same x_base it saw at training time.
# ---------------------------------------------------------------------------
def _build_base_prompt(problem_id: str) -> tuple[str, str]:
    from agent.prompts import TRITON_SYSTEM_PROMPT
    from benchmark.kernel.problems import KERNEL_PROBLEMS

    kp = KERNEL_PROBLEMS[problem_id]
    system = TRITON_SYSTEM_PROMPT
    user = (
        f"Problem: {kp.description}\n\n"
        f"Write a complete, runnable Python file that defines a function "
        f"`{kp.entry_point}(...)` with the same signature as the reference "
        f"below. It must produce output numerically close to the reference "
        f"(atol={kp.atol}, rtol={kp.rtol}) on all of these test shapes: "
        f"{kp.test_shapes}.\n\n"
        f"Reference implementation (correct but slow — replace its body with a "
        f"Triton kernel):\n```python\n{kp.reference_code}\n```\n\n"
        f"Return a single ```python fenced block containing the full "
        f"solution."
    )
    return system, user


# ---------------------------------------------------------------------------
# Reasoning-trace join (optional, future-ready)
# ---------------------------------------------------------------------------
def _prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_reasoning_index(path: Path) -> dict[str, str]:
    """Load ``reasoning_trace.jsonl`` and return {prompt_hash: reasoning}.

    Only keeps the reasoning from the *successful* attempt (i.e. the one
    that returned non-empty content — same as what evolution_trace ended
    up using).  If multiple successful attempts share a hash (shouldn't
    happen, but just in case), the last-seen wins.
    """
    out: dict[str, str] = {}
    if not path.exists():
        logger.warning("reasoning-trace file not found at %s; proceeding without", path)
        return out
    kept = 0
    with path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
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
# Per-record phase-1/phase-2 emission
# ---------------------------------------------------------------------------
def _rank_normalize(scores: list[float]) -> list[float]:
    """Return rank-percentile in [0, 1] — robust to per-problem reward scale."""
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    for new_rank, i in enumerate(order):
        ranks[i] = new_rank / max(1, n - 1)
    return ranks


def build_records(
    trace_records: list[dict[str, Any]],
    *,
    emit_phase1: bool,
    emit_phase2: bool,
    min_score: float,
    require_triton: bool,
    rank_normalize: bool,
    reasoning_index: dict[str, str],
    problem_id_override: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [r for r in trace_records if _record_is_usable(r, min_score=min_score, require_triton=require_triton)]
    logger.info("records: total=%d usable=%d", len(trace_records), len(usable))
    if not usable:
        return [], []

    # Group by problem for per-problem normalization
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for r in usable:
        pid = problem_id_override or _record_problem_id(r) or "unknown"
        by_problem.setdefault(pid, []).append(r)

    phase1_records: list[dict[str, Any]] = []
    phase2_records: list[dict[str, Any]] = []

    for pid, recs in by_problem.items():
        scores = [float(r["child_metrics"]["combined_score"]) for r in recs]
        weights = _rank_normalize(scores) if rank_normalize else scores
        logger.info(
            "problem=%s n=%d score range=[%.3f, %.3f] weight-scheme=%s",
            pid, len(recs), min(scores), max(scores),
            "rank" if rank_normalize else "raw_score",
        )
        if emit_phase2:
            try:
                base_system, base_user = _build_base_prompt(pid)
            except KeyError:
                logger.warning("no KERNEL_PROBLEMS entry for %r; skipping Phase 2 emission for it", pid)
                base_system = base_user = None

        for r, w in zip(recs, weights):
            oe_system = r["prompt"]["system"]
            oe_user = r["prompt"]["user"]
            clean_code = _clean_child_code(r["child_code"])

            # Optional reasoning trace (matches by OE messages hash — same
            # system + user message pair that was actually sent to vLLM).
            oe_messages_for_hash = [
                {"role": "system", "content": oe_system},
                {"role": "user", "content": oe_user},
            ]
            reasoning = reasoning_index.get(_prompt_hash(oe_messages_for_hash))

            cm = r["child_metrics"]
            common = {
                "problem_id": pid,
                "source_iter": r.get("iteration"),
                "source_file": r.get("_source"),
                "raw_score": float(cm["combined_score"]),
                "sample_weight": float(w),
                "correct": float(cm.get("correctness", 0.0)) >= 1.0,
                "speedup": float(cm.get("speedup", 0.0)),
            }
            if reasoning:
                common["reasoning"] = reasoning

            # Build the assistant turn.  When we have a captured
            # reasoning trace we attach it as ``thinking`` so the
            # trainer can emit a full Harmony analysis+final channel
            # sequence — teaching the model to reason-then-answer in
            # the same way it did during the OE search.  Records
            # without a reasoning trace (e.g. from the v3 pre-fix OE
            # run) only have ``content``, which trains the model to
            # emit a bare final-channel response for that sample.
            # Mixing is fine: gpt-oss natively chooses whether to use
            # the analysis channel based on reasoning_effort, so
            # exposing both kinds of targets keeps behavior flexible.
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
            if emit_phase2 and base_system is not None:
                # Reuse the same reasoning (captured on the OE prompt)
                # as the Phase-2 target, even though the prompt is
                # different.  The reasoning is about "how to solve
                # this softmax problem" — equally applicable to the
                # base prompt.  Without this, Phase 2 would be
                # content-only and the model would learn "analyze
                # when OE prompt, skip analysis when base prompt",
                # which is exactly the behavioral mismatch we want
                # to avoid.
                phase2_records.append(
                    {
                        **common,
                        "phase": "phase2",
                        "messages": [
                            {"role": "system", "content": base_system},
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="One or more evolution_trace.jsonl files.")
    parser.add_argument(
        "--reasoning-trace",
        default=None,
        help="Optional reasoning_trace.jsonl file to join by prompt hash (one per trace, or merged).",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        default="both",
        help="Which phase(s) to emit. Both emits two separate files.",
    )
    parser.add_argument("--out-phase1", default=None, help="Output path for phase-1 JSONL (required if --phase is 1 or both).")
    parser.add_argument("--out-phase2", default=None, help="Output path for phase-2 JSONL (required if --phase is 2 or both).")
    parser.add_argument("--min-score", type=float, default=0.5, help="Filter out records with combined_score below this (default 0.5).")
    parser.add_argument("--require-triton", action="store_true", default=True, help="Keep only records whose child_code actually uses triton (default True).")
    parser.add_argument("--no-require-triton", dest="require_triton", action="store_false", help="Keep records even if no triton usage detected.")
    parser.add_argument("--rank-normalize-weights", action="store_true", help="Per-problem rank-normalize sample_weight into [0,1] (robust across reward scales).")
    parser.add_argument("--problem-id-override", default=None, help="Force all records to this problem_id (skips dir-name inference).")
    args = parser.parse_args()

    emit_phase1 = args.phase in ("1", "both")
    emit_phase2 = args.phase in ("2", "both")
    if emit_phase1 and not args.out_phase1:
        parser.error("--out-phase1 is required when --phase is 1 or both")
    if emit_phase2 and not args.out_phase2:
        parser.error("--out-phase2 is required when --phase is 2 or both")

    records: list[dict[str, Any]] = []
    for path in args.traces:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        for rec in _iter_trace(p):
            rec["_source"] = str(p)
            records.append(rec)
    logger.info("loaded %d raw trace records across %d files", len(records), len(args.traces))

    reasoning_index: dict[str, str] = {}
    if args.reasoning_trace:
        reasoning_index = _load_reasoning_index(Path(args.reasoning_trace))

    phase1, phase2 = build_records(
        records,
        emit_phase1=emit_phase1,
        emit_phase2=emit_phase2,
        min_score=args.min_score,
        require_triton=args.require_triton,
        rank_normalize=args.rank_normalize_weights,
        reasoning_index=reasoning_index,
        problem_id_override=args.problem_id_override,
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

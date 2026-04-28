"""Shared prompt builder for AlgoTune base-model eval and training.

For the system message, we intentionally reuse the exact task-system
prompt from ``TaskSpec.system_message`` so base/atlas evaluations see
the same per-problem system prompt family as OE runs.

Parsing convention
------------------
Given an algotune problem id, we look at its initial_program.py:
  1. the block between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END is
     treated as the "reference (slow) implementation";
  2. the top-of-file module docstring (first triple-quoted string) is
     the problem description;
  3. the function defined inside the EVOLVE block is the entry point;
     we infer its signature by regex.

No dependency on Modal or any heavy packages — this module is imported
by both the local trainer-driver and the Modal remote eval function.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


_EVOLVE_START_RE = re.compile(r"#\s*EVOLVE-BLOCK-START")
_EVOLVE_END_RE = re.compile(r"#\s*EVOLVE-BLOCK-END")
_DEF_RE = re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)


def _extract_docstring(source: str) -> str:
    """Return the module-level docstring of ``source`` (empty if none)."""
    source = source.lstrip()
    for quote in ('"""', "'''"):
        if source.startswith(quote):
            end = source.find(quote, len(quote))
            if end != -1:
                return source[len(quote) : end].strip()
    return ""


def _extract_evolve_block(source: str) -> Optional[str]:
    """Return text between EVOLVE-BLOCK-START / -END markers (inclusive of
    the inner code, excluding the marker lines).  Returns None if the
    markers aren't present.
    """
    lines = source.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        if _EVOLVE_START_RE.search(ln):
            start = i + 1
        elif _EVOLVE_END_RE.search(ln):
            end = i
            break
    if start is None or end is None or end <= start:
        return None
    return "\n".join(lines[start:end]).strip("\n")


def _extract_entry_point(block: str) -> Optional[str]:
    m = _DEF_RE.search(block)
    return m.group(1) if m else None


def build_algotune_prompts(problem_id: str) -> tuple[str, str, str]:
    """Return ``(base_system, base_user, entry_point)`` for one algotune
    problem.

    The base prompt is intentionally short and self-contained so it
    matches the "user-facing" condition (no OE scaffolding).
    """
    # Import lazily so this file stays free of task registry side-effects
    # for static tools.
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from experiment.tasks import get_task  # noqa: E402

    spec = get_task("algotune", problem_id)
    initial = spec.initial_program
    doc = _extract_docstring(initial)
    evolve_block = _extract_evolve_block(initial) or initial
    entry_point = _extract_entry_point(evolve_block) or "<unknown>"

    user = (
        f"Problem: {problem_id}\n\n"
        f"{doc}\n\n"
        f"Reference (correct but slow) implementation — replace its body "
        f"with a faster implementation, keeping the same function name and "
        f"signature:\n"
        f"```python\n{evolve_block}\n```\n\n"
        f"Produce a single ```python fenced block containing a complete "
        f"runnable file. The entry-point function is `{entry_point}`.\n"
    )
    return spec.system_message, user, entry_point

__all__ = ["build_algotune_prompts"]

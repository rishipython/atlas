"""Helpers for extracting code answers from LLM responses.

Reasoning models (Qwen3, gpt-oss, DeepSeek-R1, ...) usually produce a long
chain-of-thought followed by a final fenced code block. We want to:

1. strip any ``<think>...</think>`` scratchpad channels that leak through,
2. pick the **last** fenced ``python`` block as the final answer
   (earlier blocks are typically exploratory / scratch),
3. fall back gracefully if the model ignored the response-format instructions.
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_PY_RE = re.compile(
    r"```(?:python|py)[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_FENCE_ANY_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(response: str) -> str:
    """Return the code portion of an LLM response.

    Strategy (first match wins):
      1. last ```python`` / ``py`` fenced block
      2. last generic ``` fenced block
      3. the raw response with <think> blocks removed
    """
    if not response:
        return ""

    cleaned = _THINK_RE.sub("", response).strip()

    matches = _FENCE_PY_RE.findall(cleaned)
    if matches:
        return matches[-1].strip()

    matches = _FENCE_ANY_RE.findall(cleaned)
    if matches:
        return matches[-1].strip()

    return cleaned

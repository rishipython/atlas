"""Atlas-wide ``sitecustomize`` — auto-imported by Python at startup.

Python's stdlib ``site`` module calls ``import sitecustomize`` during
interpreter startup, searching the full ``sys.path``.  Because our Modal
image exports ``PYTHONPATH=/atlas``, this file is imported automatically by
**every** Python process inside the container, including the spawned
``ProcessPoolExecutor`` workers that OpenEvolve uses for iteration.  That
gives us a single place to monkey-patch OpenEvolve's LLM call without
having to fork or sed-patch the wheel.

Current patches on ``OpenAILLM._call_api``:

1. Cascading retry for empty ``content`` responses (the "two-phase"
   fallback).  gpt-oss's Harmony format splits output into an analysis
   channel (reasoning) and a final channel (answer), and at
   ``reasoning_effort="high"`` (the default) the model frequently spends
   its whole output budget inside the analysis channel and emits nothing
   to the final channel, so OpenEvolve gets ``content=None`` and that
   iteration is wasted.  When we detect that, we re-issue the call with
   successively lower reasoning efforts (``medium``, then ``low``) —
   which hard-cap the analysis budget and force the model into the
   final channel — before giving up.

2. Reasoning-trace side-channel logger.  If the env var
   ``ATLAS_REASONING_TRACE_PATH`` is set, every attempt (including
   cascaded retries) is appended as a JSONL record to that file,
   capturing the full chat messages, the Harmony ``reasoning_content``
   (when vLLM is launched with ``--reasoning-parser openai_gptoss``),
   the final ``content``, attempt index / reasoning effort used, and
   usage stats.  Records are keyed by a sha256 prompt hash so they can
   be joined to OpenEvolve's ``evolution_trace.jsonl`` offline.

Nothing here is Modal-specific; locally this module is only imported if
``/atlas`` ends up on ``sys.path`` (e.g. if you ``cd /atlas && python``),
and even then the patch is a no-op unless ``openevolve`` is importable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time

try:  # openevolve is optional locally — this must not break plain imports
    from openevolve.llm.openai import OpenAILLM  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - no-op outside the Modal container
    OpenAILLM = None  # type: ignore[assignment]


# File-write lock shared across all patched calls in this process.  The
# ProcessPoolExecutor workers OpenEvolve uses each get their own
# interpreter (and hence their own lock), and we rely on the kernel's
# append-mode atomicity for line-sized writes across processes.
_REASONING_LOG_LOCK = threading.Lock()


def _prompt_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _log_reasoning(record: dict) -> None:
    """Append a single JSONL record to ``$ATLAS_REASONING_TRACE_PATH``.

    Best-effort only — any I/O error is swallowed so a logging failure
    cannot break OpenEvolve's main loop.
    """
    path = os.environ.get("ATLAS_REASONING_TRACE_PATH")
    if not path:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return
    try:
        with _REASONING_LOG_LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
    except Exception:  # noqa: BLE001 - never raise from the logger
        pass


def _install_openai_twophase_fallback() -> None:
    if OpenAILLM is None:
        return
    # Idempotent: if we've already patched, do nothing.
    if getattr(OpenAILLM, "_atlas_twophase_patched", False):
        return

    log = logging.getLogger("atlas.twophase")
    # Env-var escape hatch so CI / debugging can disable the patch without
    # rebuilding the image: ATLAS_TWOPHASE_DISABLE=1.
    if os.environ.get("ATLAS_TWOPHASE_DISABLE") == "1":
        log.info("[two-phase] disabled via ATLAS_TWOPHASE_DISABLE=1")
        return

    async def _call_api_with_cascade(self, params):
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

        loop = asyncio.get_event_loop()

        # Build the cascade of reasoning efforts to try in order. We start
        # with whatever the caller requested (typically None / "high"),
        # then fall back to progressively shorter analysis budgets. A
        # ``None`` entry means "don't pass reasoning_effort to the API",
        # which lets the model/server use its own default. Deduplicate
        # while preserving order so we never retry the same effort.
        original_effort = params.get("reasoning_effort")
        cascade: list[str | None] = []
        for effort in (original_effort, "medium", "low"):
            if effort not in cascade:
                cascade.append(effort)

        messages = params.get("messages") or []
        prompt_hash = _prompt_hash(messages) if messages else ""

        last_content = None
        for attempt_idx, effort in enumerate(cascade):
            attempt_params = dict(params)
            if effort is None:
                attempt_params.pop("reasoning_effort", None)
            else:
                attempt_params["reasoning_effort"] = effort

            response = await loop.run_in_executor(
                None,
                lambda p=attempt_params: self.client.chat.completions.create(**p),
            )
            msg = response.choices[0].message
            content = msg.content
            last_content = content

            # vLLM 0.19 renamed this field from ``reasoning_content`` to
            # ``reasoning``; some SDK versions expose it under the old
            # name, some under the new.  Also try the generic
            # ``model_dump`` route and pull any key containing
            # "reasoning" — this is resilient to further field renames.
            reasoning_content = None
            for attr_name in ("reasoning", "reasoning_content"):
                val = getattr(msg, attr_name, None)
                if isinstance(val, str) and val:
                    reasoning_content = val
                    break
            if reasoning_content is None and hasattr(msg, "model_dump"):
                try:
                    dump = msg.model_dump()
                    for k, v in dump.items():
                        if "reasoning" in k and isinstance(v, str) and v:
                            reasoning_content = v
                            break
                    # Log the full message shape once so we can see the
                    # actual field name served by this particular vLLM
                    # build (only for the first attempt of the run).
                    if not getattr(OpenAILLM, "_atlas_shape_logged", False):
                        log.warning("[two-phase] message shape: keys=%s", list(dump.keys()))
                        OpenAILLM._atlas_shape_logged = True
                except Exception:  # noqa: BLE001
                    pass
            reasoning_len = len(reasoning_content) if reasoning_content else 0

            usage_obj = getattr(response, "usage", None)
            try:
                usage = (
                    usage_obj.model_dump()
                    if usage_obj is not None and hasattr(usage_obj, "model_dump")
                    else (dict(usage_obj) if usage_obj is not None else None)
                )
            except Exception:  # noqa: BLE001
                usage = str(usage_obj) if usage_obj is not None else None

            _log_reasoning(
                {
                    "timestamp": time.time(),
                    "prompt_hash": prompt_hash,
                    "messages": messages,
                    "attempt_idx": attempt_idx,
                    "cascade_len": len(cascade),
                    "reasoning_effort": effort,
                    "original_reasoning_effort": original_effort,
                    "reasoning_content": reasoning_content,
                    "reasoning_len": reasoning_len,
                    "content": content,
                    "content_len": len(content) if content else 0,
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    "usage": usage,
                    "model": getattr(response, "model", None),
                }
            )

            if content and content.strip():
                if attempt_idx > 0:
                    log.warning(
                        "[two-phase] recovered on attempt %d/%d "
                        "with reasoning_effort=%s (prev efforts returned empty)",
                        attempt_idx + 1,
                        len(cascade),
                        effort,
                    )
                return content

            log.warning(
                "[two-phase] attempt %d/%d with reasoning_effort=%s returned empty content "
                "(reasoning_content=%d chars, usage=%s); cascading",
                attempt_idx + 1,
                len(cascade),
                effort,
                reasoning_len,
                usage,
            )

        log.error(
            "[two-phase] exhausted cascade (%s); giving up and returning last content",
            cascade,
        )
        return last_content

    OpenAILLM._call_api = _call_api_with_cascade  # type: ignore[method-assign]
    OpenAILLM._atlas_twophase_patched = True  # type: ignore[attr-defined]


_install_openai_twophase_fallback()

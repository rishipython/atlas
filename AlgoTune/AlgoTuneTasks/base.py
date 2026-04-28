"""Minimal task registry compatible with the vendored AlgoTune evaluators."""

from __future__ import annotations

from typing import Callable

TASK_REGISTRY: dict[str, type] = {}


class Task:
    """Lightweight base class placeholder."""


def register_task(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


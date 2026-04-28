"""Task registry for the trimmed AlgoTune-only experiment."""

from __future__ import annotations

from .algotune import FAMILY as ALGOTUNE_FAMILY
from .base import TaskFamily, TaskSpec

TASK_FAMILIES: dict[str, TaskFamily] = {
    ALGOTUNE_FAMILY.name: ALGOTUNE_FAMILY,
}


def get_task(family: str, problem_id: str) -> TaskSpec:
    if family not in TASK_FAMILIES:
        raise ValueError(
            f"unknown task family {family!r}; available: {list(TASK_FAMILIES)}"
        )
    return TASK_FAMILIES[family].make_task(problem_id)


def list_families() -> dict[str, list[str]]:
    return {name: fam.available_problems for name, fam in TASK_FAMILIES.items()}


__all__ = [
    "TaskFamily",
    "TaskSpec",
    "TASK_FAMILIES",
    "get_task",
    "list_families",
]

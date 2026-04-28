"""Task abstraction for the AlgoTune-only OpenEvolve runner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    task_family: str
    """Task family name. This branch only keeps ``"algotune"``."""

    problem_id: str
    """Task identifier within the family."""

    initial_program: str
    """Full source of ``initial_program.py``."""

    evaluator: str
    """Full source of ``evaluator.py``."""

    system_message: str
    """OpenEvolve system prompt for this task."""

    extra_packages: list[str] = field(default_factory=list)
    """Unused in the current branch but kept for simple task metadata."""

    evaluator_timeout: int = 300
    """Per-candidate evaluator timeout in seconds."""

    uses_vllm_in_evaluator: bool = False
    """Whether the evaluator calls back into vLLM. AlgoTune tasks do not."""


@dataclass
class TaskFamily:
    """Registered task family.  ``make_task`` returns a ``TaskSpec``."""

    name: str
    make_task: callable  # (problem_id: str) -> TaskSpec
    available_problems: list[str]

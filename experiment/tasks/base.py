"""Task abstraction for the OpenEvolve runner.

A ``TaskSpec`` bundles everything OpenEvolve needs to run on a given
problem:
  - the initial program source (the code OpenEvolve mutates)
  - the evaluator source (the ``evaluate(program_path)`` module OE imports)
  - the system message OE injects into every mutation prompt
  - optional hints the runner uses to pick vLLM / eval parameters

Task families register a factory function ``make_task(problem_id) ->
TaskSpec`` and are plugged into the runner via the ``TASK_FAMILIES``
dispatch dict in ``experiment.tasks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    task_family: str
    """Name of the task family (``kernel``, ``alphaevolve``, ``algotune``,
    ``prompt_opt``).  Used to namespace run outputs."""

    problem_id: str
    """Task-family-specific identifier (``softmax``, ``circle_packing_26``,
    ``pairwise_distance``, ``gsm8k_3q``)."""

    initial_program: str
    """Full source of ``initial_program.py``.  Must contain at least one
    EVOLVE-BLOCK-START / EVOLVE-BLOCK-END pair."""

    evaluator: str
    """Full source of ``evaluator.py``.  Must define ``evaluate(program_path)
    -> dict`` returning at minimum ``{"combined_score": float}``."""

    system_message: str
    """OpenEvolve system prompt.  Injected into every LLM mutation
    request."""

    extra_packages: list[str] = field(default_factory=list)
    """pip packages required by this task's evaluator that are NOT in the
    base image.  The runner installs them before each run."""

    evaluator_timeout: int = 300
    """Per-candidate evaluator timeout in seconds (OE's ``evaluator.timeout``
    config key).  Tasks with cheap evals (e.g. circle packing, pairwise
    distance) should use a small number; tasks that call the LLM in-loop
    (prompt_opt) need a bigger budget."""

    uses_vllm_in_evaluator: bool = False
    """Whether the evaluator makes HTTP calls to the local vLLM server.
    ``prompt_opt`` does; the others don't.  The runner uses this to decide
    ``parallel_evaluations`` (keep at 1 when the evaluator contends with
    OE's own LLM calls)."""


@dataclass
class TaskFamily:
    """Registered task family.  ``make_task`` returns a ``TaskSpec``."""

    name: str
    make_task: callable  # (problem_id: str) -> TaskSpec
    available_problems: list[str]

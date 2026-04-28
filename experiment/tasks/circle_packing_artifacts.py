"""Circle-packing-with-artifacts task family.

Single-problem family vendoring OE's ``circle_packing_with_artifacts``
example — pack 26 circles in the unit square, but with the OE artifact
side-channel enabled so every candidate emits a machine-readable
diagnostic alongside its scalar score. Useful when we want to generate a
richer training signal (failure modes, partial constraints violated, per
circle diagnostics).

Problem ID: ``circle_packing_26_artifacts``.
"""

from __future__ import annotations

from pathlib import Path

from .alphaevolve_math import _parse_evaluator_timeout, _parse_yaml_system_message
from .base import TaskFamily, TaskSpec

_PROBLEM_DIR = (
    Path(__file__).parent / "_oe_problems" / "circle_packing_artifacts"
)


def make_task(problem_id: str) -> TaskSpec:
    assert problem_id == "circle_packing_26_artifacts", (
        f"circle_packing_artifacts only supports 'circle_packing_26_artifacts'; "
        f"got {problem_id!r}"
    )
    initial_program = (_PROBLEM_DIR / "initial_program.py").read_text()
    evaluator = (_PROBLEM_DIR / "evaluator.py").read_text()
    config_yaml = (_PROBLEM_DIR / "config.yaml").read_text()
    system_message = _parse_yaml_system_message(config_yaml)
    timeout = _parse_evaluator_timeout(config_yaml, default=300)
    return TaskSpec(
        task_family="circle_packing_artifacts",
        problem_id=problem_id,
        initial_program=initial_program,
        evaluator=evaluator,
        system_message=system_message,
        extra_packages=["scipy>=1.11"],
        evaluator_timeout=timeout,
        uses_vllm_in_evaluator=False,
    )


FAMILY = TaskFamily(
    name="circle_packing_artifacts",
    make_task=make_task,
    available_problems=["circle_packing_26_artifacts"],
)

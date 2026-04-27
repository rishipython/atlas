"""AlgoTune task family using upstream OpenEvolve example assets.

This family loads the selected AlgoTune example problems from
``experiment/tasks/_oe_problems/algotune_examples`` and exposes them
through our native ``TaskSpec`` registry.
"""

from __future__ import annotations

from pathlib import Path

from .base import TaskFamily, TaskSpec

_THIS_DIR = Path(__file__).resolve().parent
_ALGOTUNE_ROOT = _THIS_DIR / "_oe_problems" / "algotune_examples"

# Chosen sample of relatively simple problems for quick experiments.
SIMPLE_SAMPLE_PROBLEMS: list[str] = [
    "fft_convolution",
    "convolve2d_full_fill",
    "affine_transform_2d",
]


def _extract_system_prompt(config_text: str) -> str:
    lines = config_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("  system_message: |"):
            start = i + 1
            break
    if start is None:
        raise ValueError("system_message block not found in config.yaml")

    out: list[str] = []
    for j in range(start, len(lines)):
        ln = lines[j]
        if ln.startswith("  ") and not ln.startswith("    "):
            break
        if ln.startswith("    "):
            out.append(ln[4:])
        elif ln == "":
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _problem_dir(problem_id: str) -> Path:
    d = _ALGOTUNE_ROOT / problem_id
    if not d.exists():
        raise ValueError(
            f"algotune has no problem {problem_id!r}; "
            f"available: {sorted(SIMPLE_SAMPLE_PROBLEMS)}"
        )
    return d


def make_task(problem_id: str) -> TaskSpec:
    if problem_id not in SIMPLE_SAMPLE_PROBLEMS:
        raise ValueError(
            f"algotune has no problem {problem_id!r}; "
            f"available: {sorted(SIMPLE_SAMPLE_PROBLEMS)}"
        )

    d = _problem_dir(problem_id)
    initial = (d / "initial_program.py").read_text()
    evaluator = (d / "evaluator.py").read_text()
    config_text = (d / "config.yaml").read_text()
    system_message = _extract_system_prompt(config_text)

    return TaskSpec(
        task_family="algotune",
        problem_id=problem_id,
        initial_program=initial,
        evaluator=evaluator,
        system_message=system_message,
        extra_packages=[],
        evaluator_timeout=200,
        uses_vllm_in_evaluator=False,
    )


FAMILY = TaskFamily(
    name="algotune",
    make_task=make_task,
    available_problems=sorted(SIMPLE_SAMPLE_PROBLEMS),
)


__all__ = ["FAMILY", "make_task", "SIMPLE_SAMPLE_PROBLEMS"]

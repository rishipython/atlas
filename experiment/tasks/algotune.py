"""AlgoTune task family backed by vendored OpenEvolve example assets."""

from __future__ import annotations

from pathlib import Path

from .base import TaskFamily, TaskSpec

_THIS_DIR = Path(__file__).resolve().parent
_ALGOTUNE_ROOT = _THIS_DIR / "_oe_problems" / "algotune_examples"

def _available_problem_ids() -> list[str]:
    problems: list[str] = []
    for child in sorted(_ALGOTUNE_ROOT.iterdir()):
        if not child.is_dir():
            continue
        required = ["initial_program.py", "evaluator.py", "config.yaml"]
        if all((child / name).exists() for name in required):
            problems.append(child.name)
    return problems


AVAILABLE_PROBLEMS = _available_problem_ids()


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
            f"available: {AVAILABLE_PROBLEMS}"
        )
    return d


def make_task(problem_id: str) -> TaskSpec:
    if problem_id not in AVAILABLE_PROBLEMS:
        raise ValueError(
            f"algotune has no problem {problem_id!r}; "
            f"available: {AVAILABLE_PROBLEMS}"
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
    available_problems=AVAILABLE_PROBLEMS,
)


__all__ = ["AVAILABLE_PROBLEMS", "FAMILY", "make_task"]

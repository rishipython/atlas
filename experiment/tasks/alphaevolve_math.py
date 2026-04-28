"""AlphaEvolve math-problems task family — directory-based loader.

The OpenEvolve project ships a set of ready-to-run AlphaEvolve problems in
``experiment/tasks/_oe_problems/alphaevolve_math/`` (vendored from
``codelion/openevolve/examples/alphaevolve_math_problems``). Each problem
dir contains:

  - ``initial_program.py``  (the seed code OE evolves; contains
    ``EVOLVE-BLOCK-START``/``-END`` markers)
  - ``evaluator.py``        (defines ``evaluate(program_path) -> dict``)
  - ``config.yaml``         (OE config; we only mine ``prompt.system_message``
    and ``evaluator.timeout`` from this)
  - ``requirements.txt``    (pip packages the evaluator / initial_program
    need)

A few problems have size variants and are nested one level deeper (e.g.
``hexagon_packing/11`` and ``hexagon_packing/12``). This loader flattens
those variants into distinct ``problem_id``s of the form
``<problem>_<variant>`` (e.g. ``hexagon_packing_11``).

Problem IDs currently exposed:
  - ``first_autocorr_ineq``, ``second_autocorr_ineq``, ``third_autocorr_ineq``
  - ``circle_packing_rect``, ``erdos_min_overlap``, ``matmul``
  - ``kissing_number``, ``sums_diffs_finite_sets``, ``uncertainty_ineq``
  - ``heilbronn_triangle``
  - ``hexagon_packing_11``, ``hexagon_packing_12``
  - ``heilbronn_convex_13``, ``heilbronn_convex_14``
  - ``minimizing_max_min_dist_2``, ``minimizing_max_min_dist_3``

All evaluators maximize ``combined_score`` (per the OE convention); some
raw metrics (e.g. ``outer_hex_side_length``) are minimized, but the
``combined_score`` is always a maximization objective.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import TaskFamily, TaskSpec

_PROBLEMS_ROOT = Path(__file__).parent / "_oe_problems" / "alphaevolve_math"

# Which AlphaEvolve dirs have numeric size/variant subdirs and what their
# ``problem_id`` suffixes should be.
_NESTED_VARIANTS: dict[str, list[str]] = {
    "hexagon_packing": ["11", "12"],
    "heilbronn_convex": ["13", "14"],
    "minimizing_max_min_dist": ["2", "3"],
}

# Shared union of pip packages the evaluators + initial_program files may
# import. We install all of them so we don't have to track which problem
# needs what.
_EXTRA_PACKAGES = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "optax>=0.1.7",
    "sympy>=1.12",
    "tqdm>=4.66",
    "matplotlib>=3.7",
]


@dataclass
class _ProblemFiles:
    initial_program: str
    evaluator: str
    system_message: str
    evaluator_timeout: int


def _parse_yaml_system_message(config_yaml: str) -> str:
    """Extract ``prompt.system_message`` from a config.yaml string.

    We avoid pulling in PyYAML for such a narrow task — the format is a
    stable YAML multi-line block under a ``prompt:`` → ``system_message:
    |`` header. We lift it out by line-based heuristics.
    """
    lines = config_yaml.splitlines()
    in_prompt = False
    in_system = False
    indent: Optional[int] = None
    out_lines: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not in_prompt:
            if stripped.startswith("prompt:"):
                in_prompt = True
            continue
        # We're inside prompt: — an unindented key ends the prompt block
        if line and not line[0].isspace() and not stripped.startswith("#"):
            break
        if not in_system:
            # look for `system_message: |`
            m = re.match(r"(\s+)system_message:\s*\|", line)
            if m:
                in_system = True
            continue
        # We're inside the system_message literal block.
        # Lines belonging to the block are indented at least one more level
        # than the ``system_message:`` header. We capture the indent from
        # the first content line and stop when indent decreases.
        if indent is None:
            if stripped == "":
                out_lines.append("")
                continue
            indent = len(line) - len(stripped)
        current_indent = len(line) - len(line.lstrip())
        if stripped == "":
            out_lines.append("")
            continue
        if current_indent < indent:
            break
        out_lines.append(line[indent:])

    # Strip trailing blank lines.
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    if not out_lines:
        raise ValueError("could not locate prompt.system_message in config.yaml")
    return "\n".join(out_lines)


def _parse_evaluator_timeout(config_yaml: str, default: int = 360) -> int:
    """Best-effort scrape of ``evaluator.timeout`` (seconds)."""
    lines = config_yaml.splitlines()
    in_evaluator = False
    for line in lines:
        stripped = line.lstrip()
        if line and not line[0].isspace() and not stripped.startswith("#"):
            in_evaluator = stripped.startswith("evaluator:")
            continue
        if in_evaluator:
            m = re.match(r"\s+timeout:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return default


def _resolve_problem_dir(problem_id: str) -> tuple[Path, str]:
    """Map a flattened problem_id to a directory on disk.

    Returns ``(problem_dir, canonical_family_name)``; the canonical name
    is used for namespacing (currently always ``"alphaevolve_math"``).
    """
    # Nested variants: <name>_<variant> -> _PROBLEMS_ROOT/<name>/<variant>
    for base, variants in _NESTED_VARIANTS.items():
        for v in variants:
            if problem_id == f"{base}_{v}":
                d = _PROBLEMS_ROOT / base / v
                if not d.is_dir():
                    raise FileNotFoundError(f"{d} not found")
                return d, "alphaevolve_math"
    # Flat problems
    d = _PROBLEMS_ROOT / problem_id
    if d.is_dir() and (d / "initial_program.py").is_file():
        return d, "alphaevolve_math"
    raise ValueError(
        f"unknown alphaevolve_math problem_id {problem_id!r}; "
        f"available: {_list_available_problems()}"
    )


def _load_problem(problem_id: str) -> _ProblemFiles:
    problem_dir, _ = _resolve_problem_dir(problem_id)
    initial_program = (problem_dir / "initial_program.py").read_text()
    evaluator = (problem_dir / "evaluator.py").read_text()
    config_yaml = (problem_dir / "config.yaml").read_text()
    system_message = _parse_yaml_system_message(config_yaml)
    timeout = _parse_evaluator_timeout(config_yaml, default=360)
    return _ProblemFiles(
        initial_program=initial_program,
        evaluator=evaluator,
        system_message=system_message,
        evaluator_timeout=timeout,
    )


def _list_available_problems() -> list[str]:
    available: list[str] = []
    if not _PROBLEMS_ROOT.is_dir():
        return available
    for d in sorted(_PROBLEMS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name in _NESTED_VARIANTS:
            for v in _NESTED_VARIANTS[name]:
                variant_dir = d / v
                if variant_dir.is_dir() and (variant_dir / "initial_program.py").is_file():
                    available.append(f"{name}_{v}")
            continue
        if (d / "initial_program.py").is_file():
            available.append(name)
    return available


def make_task(problem_id: str) -> TaskSpec:
    loaded = _load_problem(problem_id)
    return TaskSpec(
        task_family="alphaevolve_math",
        problem_id=problem_id,
        initial_program=loaded.initial_program,
        evaluator=loaded.evaluator,
        system_message=loaded.system_message,
        extra_packages=_EXTRA_PACKAGES,
        evaluator_timeout=loaded.evaluator_timeout,
        uses_vllm_in_evaluator=False,
    )


FAMILY = TaskFamily(
    name="alphaevolve_math",
    make_task=make_task,
    available_problems=_list_available_problems(),
)

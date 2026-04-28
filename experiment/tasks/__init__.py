"""Task registry for the OpenEvolve runner.

Every task family is a module that defines:
  - ``make_task(problem_id: str) -> TaskSpec``
  - ``FAMILY`` of type ``TaskFamily``

Call ``get_task(family, problem_id)`` to build a ``TaskSpec`` for any
registered family, or ``list_families()`` to enumerate them.

Families currently registered:

- ``kernel``                   — Triton kernel speedup (softmax, layernorm, matmul, ...)
- ``alphaevolve``              — legacy single-problem alphaevolve family
                                 (``circle_packing_26``), kept for
                                 reproducibility with earlier OE runs
- ``alphaevolve_math``         — full AlphaEvolve math-problem bank
                                 vendored from ``codelion/openevolve``
                                 (13 problems, 16 variants)
- ``circle_packing_artifacts`` — OE circle-packing with artifact channel
- ``algotune``                 — upstream OpenEvolve AlgoTune problems
                                 (fft_convolution,
                                 convolve2d_full_fill,
                                 affine_transform_2d,
                                 fft_cmplx_scipy_fftpack,
                                 lu_factorization,
                                 polynomial_real,
                                 eigenvectors_complex,
                                 psd_cone_projection)
- ``prompt_opt``               — prompt optimisation on GSM8K-style
                                 problems (faded; evaluator saturates)
"""

from __future__ import annotations

from .algotune import FAMILY as ALGOTUNE_FAMILY
from .alphaevolve_circle_packing import FAMILY as ALPHAEVOLVE_FAMILY
from .alphaevolve_math import FAMILY as ALPHAEVOLVE_MATH_FAMILY
from .base import TaskFamily, TaskSpec
from .circle_packing_artifacts import FAMILY as CIRCLE_PACKING_ARTIFACTS_FAMILY
from .kernel import FAMILY as KERNEL_FAMILY
from .prompt_opt_gsm8k import FAMILY as PROMPT_OPT_FAMILY

TASK_FAMILIES: dict[str, TaskFamily] = {
    KERNEL_FAMILY.name: KERNEL_FAMILY,
    ALPHAEVOLVE_FAMILY.name: ALPHAEVOLVE_FAMILY,
    ALPHAEVOLVE_MATH_FAMILY.name: ALPHAEVOLVE_MATH_FAMILY,
    CIRCLE_PACKING_ARTIFACTS_FAMILY.name: CIRCLE_PACKING_ARTIFACTS_FAMILY,
    ALGOTUNE_FAMILY.name: ALGOTUNE_FAMILY,
    PROMPT_OPT_FAMILY.name: PROMPT_OPT_FAMILY,
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

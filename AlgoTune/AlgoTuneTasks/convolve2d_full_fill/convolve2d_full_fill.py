from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.convolve2d_full_fill.initial_program import (
    Convolve2DFullFill as _Convolve2DFullFill,
)


@register_task("convolve2d_full_fill")
class Convolve2DFullFill(_Convolve2DFullFill):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        a = rng.normal(size=(30 * n, 30 * n)).astype(np.float64)
        b = rng.normal(size=(8 * n, 8 * n)).astype(np.float64)
        return a, b


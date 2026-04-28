from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.polynomial_real.initial_program import (
    PolynomialReal as _PolynomialReal,
)


@register_task("polynomial_real")
class PolynomialReal(_PolynomialReal):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        roots = rng.uniform(-1.0, 1.0, size=max(1, n))
        coeffs = np.poly(roots).real.astype(np.float64)
        return coeffs.tolist()


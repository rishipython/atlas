from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.lu_factorization.initial_program import (
    LUFactorization as _LUFactorization,
)


@register_task("lu_factorization")
class LUFactorization(_LUFactorization):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        a = rng.normal(size=(n, n)).astype(np.float64)
        a += np.eye(n, dtype=np.float64) * max(1.0, n * 0.1)
        return {"matrix": a}


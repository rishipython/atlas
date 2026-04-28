from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.eigenvectors_complex.initial_program import (
    EigenvectorsComplex as _EigenvectorsComplex,
)


@register_task("eigenvectors_complex")
class EigenvectorsComplex(_EigenvectorsComplex):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        return rng.normal(size=(n, n)).astype(np.float64)


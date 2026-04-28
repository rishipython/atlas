from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.psd_cone_projection.initial_program import (
    PSDConeProjection as _PSDConeProjection,
)


@register_task("psd_cone_projection")
class PSDConeProjection(_PSDConeProjection):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        a = rng.normal(size=(n, n)).astype(np.float64)
        a = (a + a.T) / 2.0
        return {"A": a}

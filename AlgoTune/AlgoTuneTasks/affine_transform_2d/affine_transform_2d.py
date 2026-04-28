from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.affine_transform_2d.initial_program import (
    AffineTransform2D as _AffineTransform2D,
)


@register_task("affine_transform_2d")
class AffineTransform2D(_AffineTransform2D):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        image = rng.uniform(0.0, 255.0, size=(n, n)).astype(np.float64)
        theta = rng.uniform(-0.5, 0.5)
        scale_x = rng.uniform(0.8, 1.2)
        scale_y = rng.uniform(0.8, 1.2)
        shear = rng.uniform(-0.15, 0.15)
        tx = rng.uniform(-0.1 * n, 0.1 * n)
        ty = rng.uniform(-0.1 * n, 0.1 * n)
        matrix = np.array(
            [
                [scale_x * np.cos(theta), -np.sin(theta) + shear, tx],
                [np.sin(theta), scale_y * np.cos(theta), ty],
            ],
            dtype=np.float64,
        )
        return {"image": image, "matrix": matrix}


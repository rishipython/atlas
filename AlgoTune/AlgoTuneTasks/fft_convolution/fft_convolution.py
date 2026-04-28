from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.fft_convolution.initial_program import (
    FFTConvolution as _FFTConvolution,
)


@register_task("fft_convolution")
class FFTConvolution(_FFTConvolution):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        x_len = max(8, n)
        y_len = max(4, n // 2)
        mode = ["full", "same", "valid"][random_seed % 3]
        return {
            "signal_x": rng.normal(size=x_len).astype(np.float64).tolist(),
            "signal_y": rng.normal(size=y_len).astype(np.float64).tolist(),
            "mode": mode,
        }


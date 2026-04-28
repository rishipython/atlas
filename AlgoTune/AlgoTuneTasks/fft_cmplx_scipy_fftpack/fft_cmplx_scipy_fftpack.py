from __future__ import annotations

import numpy as np

from AlgoTuneTasks.base import register_task
from experiment.tasks._oe_problems.algotune_examples.fft_cmplx_scipy_fftpack.initial_program import (
    FFTComplexScipyFFTpack as _FFTComplexScipyFFTpack,
)


@register_task("fft_cmplx_scipy_fftpack")
class FFTComplexScipyFFTpack(_FFTComplexScipyFFTpack):
    def generate_problem(self, n: int, random_seed: int = 0):
        rng = np.random.default_rng(random_seed)
        real = rng.normal(size=(n, n))
        imag = rng.normal(size=(n, n))
        return (real + 1j * imag).astype(np.complex128)


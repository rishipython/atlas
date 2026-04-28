import numpy as np
import scipy.linalg

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to :func:`scipy.linalg.expm`, which
    uses a scaling‑and‑squaring algorithm with Pade approximants for
    high performance and numerical stability.
    """
    return scipy.linalg.expm(A)
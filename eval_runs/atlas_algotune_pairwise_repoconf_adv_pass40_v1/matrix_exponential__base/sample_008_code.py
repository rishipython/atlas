import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to :func:`scipy.linalg.expm`, which
    provides a highly optimized, numerically stable computation
    that matches SciPy's reference implementation.
    """
    return expm(A)
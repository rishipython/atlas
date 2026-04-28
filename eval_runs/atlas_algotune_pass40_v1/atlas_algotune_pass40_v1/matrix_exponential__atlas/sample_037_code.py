import numpy as np
from scipy.linalg import expm as _scipy_expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to scipy.linalg.expm, which
    uses a highly optimized scaling-and-squaring algorithm with
    Padé approximants. The result is numerically stable and
    matches scipy's output to machine precision.
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    return _scipy_expm(A)
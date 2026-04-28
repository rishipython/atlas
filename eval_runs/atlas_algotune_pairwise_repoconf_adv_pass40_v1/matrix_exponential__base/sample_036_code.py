import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's highly optimized
    ``scipy.linalg.expm`` function.
    """
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
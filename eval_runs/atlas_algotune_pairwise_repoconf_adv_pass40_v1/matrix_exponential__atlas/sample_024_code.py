import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's fast and accurate
    ``scipy.linalg.expm`` routine.
    """
    return scipy.linalg.expm(A)
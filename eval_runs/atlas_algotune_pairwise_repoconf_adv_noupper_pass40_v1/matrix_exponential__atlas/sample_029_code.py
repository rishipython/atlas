import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to SciPy's highly optimized
    ``scipy.linalg.expm`` routine, which uses a scaling-and-squaring
    algorithm with Padé approximants.  It is orders of magnitude faster
    than the naive Taylor series and matches the reference implementation
    within the typical numerical tolerances.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64.

    Returns
    -------
    np.ndarray
        The matrix exponential of ``A``.
    """
    return expm(A)
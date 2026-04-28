import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to SciPy's highly optimized routine
    ``scipy.linalg.expm``.  It is numerically stable and matches
    ``scipy.linalg.expm`` within machine precision.

    Parameters
    ----------
    A : np.ndarray
        A square matrix of shape (n, n) with dtype float64 (or convertible).

    Returns
    -------
    expm_A : np.ndarray
        The matrix exponential of ``A``.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square")
    # Ensure we are working with float64 for consistency with SciPy
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
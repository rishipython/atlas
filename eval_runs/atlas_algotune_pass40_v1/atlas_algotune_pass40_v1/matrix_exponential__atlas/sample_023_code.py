import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's efficient
    `scipy.linalg.expm`, which is numerically stable and fast.

    Parameters
    ----------
    A : np.ndarray
        A square matrix of shape (n, n) with dtype float64.

    Returns
    -------
    np.ndarray
        The matrix exponential of ``A``.
    """
    return expm(A)
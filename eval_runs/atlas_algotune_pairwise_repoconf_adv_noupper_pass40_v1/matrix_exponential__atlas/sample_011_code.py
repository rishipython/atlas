import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64.

    Returns
    -------
    np.ndarray
        The matrix exponential exp(A), computed using SciPy's
        highly optimized implementation (scaling and squaring with
        Padé approximants).
    """
    # SciPy's expm already handles scaling/squaring and Padé
    # approximants, providing a fast and numerically stable result.
    return expm(A)
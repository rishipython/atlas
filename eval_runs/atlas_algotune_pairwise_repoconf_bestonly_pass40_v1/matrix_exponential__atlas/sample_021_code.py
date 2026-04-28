import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) and dtype float64.

    Returns
    -------
    np.ndarray
        The matrix exponential exp(A).
    """
    return expm(A)
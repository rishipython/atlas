import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's optimized routine,
    which uses scaling-and-squaring with Padé approximants for
    high performance and numerical stability.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64.

    Returns
    -------
    result : np.ndarray
        The matrix exponential exp(A), of shape (n, n) and dtype float64.
    """
    # Ensure input is a NumPy array of float64
    A = np.asarray(A, dtype=np.float64)

    # Validate that A is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")

    # Use SciPy's highly optimized implementation
    return expm(A)
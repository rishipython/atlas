import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to ``scipy.linalg.expm`` for
    speed and numerical stability.

    Parameters
    ----------
    A : np.ndarray
        A square matrix of shape (n, n) with dtype float64.

    Returns
    -------
    np.ndarray
        The matrix exponential expm(A).
    """
    # Ensure input is a NumPy array of float64
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    return scipy.linalg.expm(A)
import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix A.

    This implementation delegates to scipy.linalg.expm, which
    uses an efficient scaling and squaring algorithm with Pade
    approximants. It matches scipy.linalg.expm(A) to within the
    default tolerances used by SciPy.

    Parameters
    ----------
    A : np.ndarray
        Square array of shape (n, n) with dtype float64 (or convertible to it).

    Returns
    -------
    np.ndarray
        The matrix exponential of A.
    """
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
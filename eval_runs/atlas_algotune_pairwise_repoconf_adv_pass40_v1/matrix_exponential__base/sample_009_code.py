import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's highly optimized
    `scipy.linalg.expm`, which uses a Padé approximation with
    scaling and squaring.  It is both fast and numerically stable.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64.

    Returns
    -------
    np.ndarray
        The matrix exponential of ``A``.
    """
    # Validate shape and dtype
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if A.dtype != np.float64:
        A = A.astype(np.float64, copy=False)

    return expm(A)
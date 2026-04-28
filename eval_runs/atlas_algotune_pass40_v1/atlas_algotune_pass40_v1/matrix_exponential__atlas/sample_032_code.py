import numpy as np
import scipy.linalg as la


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's highly optimized
    expm routine, which uses scaling and squaring with a Pade
    approximant and is both fast and numerically stable.
    """
    # Validate input
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    n, m = A.shape
    if n != m:
        raise ValueError("Input must be a square matrix.")
    if A.dtype != np.float64:
        # SciPy's expm accepts any numeric dtype, but we keep
        # the result dtype consistent with the reference.
        A = A.astype(np.float64, copy=False)

    return la.expm(A)
import numpy as np
from scipy.linalg import expm as _scipy_expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's highly optimized expm routine.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64.
    
    Returns
    -------
    expA : np.ndarray
        The matrix exponential of ``A``.
    """
    # SciPy's expm uses a Padé approximation with scaling and squaring,
    # which is both fast and numerically stable for a wide range of inputs.
    return _scipy_expm(A)
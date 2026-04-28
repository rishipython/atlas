import numpy as np
from scipy.linalg import expm as _scipy_expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    
    This implementation delegates to SciPy's highly-optimized
    `scipy.linalg.expm`, which uses a scaling-and-squaring
    algorithm with a Padé approximant.  It is numerically stable
    and significantly faster than a naive Taylor series.
    
    Parameters
    ----------
    A : np.ndarray
        Square array of shape (n, n) with dtype float64.
    
    Returns
    -------
    np.ndarray
        The matrix exponential of ``A``.
    """
    return _scipy_expm(A)
import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    
    This implementation delegates to SciPy's highly optimized
    `scipy.linalg.expm`, which uses a Padé approximant with scaling
    and squaring for numerical stability and speed.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64.
    
    Returns
    -------
    np.ndarray
        The matrix exponential exp(A).
    """
    return scipy.linalg.expm(A)
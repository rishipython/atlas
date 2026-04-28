import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    
    This implementation delegates to SciPy's highly optimized expm
    routine, which uses scaling and squaring with a Padé approximant
    for numerical stability and speed.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n) with dtype float64 (or convertible to float64).
    
    Returns
    -------
    np.ndarray
        The matrix exponential of A.
    """
    # Ensure the input is a float64 array
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
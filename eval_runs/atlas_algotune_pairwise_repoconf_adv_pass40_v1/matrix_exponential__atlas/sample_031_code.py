import numpy as np
from scipy.linalg import expm as _scipy_expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    
    This implementation simply delegates to SciPy's highly optimized
    expm routine, which uses a scaling-and-squaring algorithm with a
    Pade approximant. The result matches ``scipy.linalg.expm(A)`` to
    machine precision.
    
    Parameters
    ----------
    A : np.ndarray
        A square matrix of dtype float64.
    
    Returns
    -------
    np.ndarray
        The matrix exponential of ``A``.
    """
    return _scipy_expm(A)
import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    
    This implementation delegates to SciPy's highly optimized
    ``scipy.linalg.expm`` routine, which uses scaling and squaring
    with Pade approximants and is numerically stable for a wide
    range of inputs.
    """
    # Ensure the input is of type float64 for consistency with the reference
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
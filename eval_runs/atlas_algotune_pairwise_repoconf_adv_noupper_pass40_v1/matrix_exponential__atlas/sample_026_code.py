import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    Uses SciPy's highly optimized implementation based on scaling and
    squaring with Pade approximants, which matches scipy.linalg.expm
    within machine precision.
    """
    # Ensure the input is a float64 array
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's efficient implementation
    which uses a scaling and squaring algorithm with Padé approximants.
    """
    return expm(A)
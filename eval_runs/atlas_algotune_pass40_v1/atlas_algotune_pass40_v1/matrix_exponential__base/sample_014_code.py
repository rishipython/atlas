import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    
    This implementation delegates to :func:`scipy.linalg.expm`, which
    uses a robust scaling-and-squaring algorithm with Pade approximants.
    It is both fast and numerically stable, matching the reference
    implementation within the tolerances used by the tests.
    """
    return expm(A)
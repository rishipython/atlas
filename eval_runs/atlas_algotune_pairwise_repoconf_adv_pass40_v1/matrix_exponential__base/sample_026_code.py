import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # ``scipy.linalg.expm`` implements a highly optimized scaling-and-squaring
    # algorithm with Pade approximants, which is both fast and numerically
    # stable.  It matches the reference implementation within tight
    # tolerances.
    return expm(A)
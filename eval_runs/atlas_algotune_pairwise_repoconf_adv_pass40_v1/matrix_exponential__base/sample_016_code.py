import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    A = np.asarray(A, dtype=np.float64)
    n, m = A.shape
    if n != m:
        raise ValueError("Input matrix must be square")
    return expm(A)
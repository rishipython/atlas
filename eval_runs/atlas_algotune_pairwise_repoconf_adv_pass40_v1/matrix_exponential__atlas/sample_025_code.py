import numpy as np
from scipy.linalg import expm as scipy_expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Validate input shape and dtype
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if A.dtype != np.float64:
        A = A.astype(np.float64)

    # Use SciPy's highly optimized implementation
    return scipy_expm(A)
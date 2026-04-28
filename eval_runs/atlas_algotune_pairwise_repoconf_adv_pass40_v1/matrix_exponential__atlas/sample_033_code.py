import numpy as np
from scipy.linalg import expm as scipy_expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Delegate to the highly optimized SciPy implementation
    return scipy_expm(A)
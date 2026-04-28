import numpy as np
from scipy.linalg import expm as _expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Delegate to SciPy's highly optimized implementation.
    return _expm(A)
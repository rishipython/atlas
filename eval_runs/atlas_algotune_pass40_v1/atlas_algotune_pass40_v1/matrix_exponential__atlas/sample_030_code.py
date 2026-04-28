import numpy as np
from scipy.linalg import expm as _expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # SciPy's implementation uses scaling & squaring with Pade approximants
    # and is highly optimized and numerically stable.
    return _expm(A)
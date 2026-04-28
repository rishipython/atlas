import numpy as np
from scipy.linalg import expm as _scipy_expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Ensure input is a NumPy array of type float64
    A = np.asarray(A, dtype=np.float64)
    # Use SciPy's highly optimized implementation (Pade + scaling/squaring)
    return _scipy_expm(A)
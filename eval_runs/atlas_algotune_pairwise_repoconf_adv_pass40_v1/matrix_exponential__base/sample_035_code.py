import numpy as np
from scipy.linalg import expm as _scipy_expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Ensure the input is a NumPy array with dtype float64
    A = np.asarray(A, dtype=np.float64)
    # Use SciPy's fast and numerically stable implementation
    return _scipy_expm(A)
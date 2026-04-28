import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Ensure the input is a NumPy array of float64
    A = np.asarray(A, dtype=np.float64)
    # Delegate to SciPy's fast and accurate implementation
    return scipy.linalg.expm(A)
import numpy as np
import scipy.linalg

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Ensure A is a NumPy array of type float64
    A = np.asarray(A, dtype=np.float64)
    # Use SciPy's highly optimized and numerically stable implementation
    return scipy.linalg.expm(A)
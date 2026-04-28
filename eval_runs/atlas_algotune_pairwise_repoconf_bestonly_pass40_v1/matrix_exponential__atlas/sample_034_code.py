import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's highly optimized
    ``scipy.linalg.expm`` function, which uses a Padé approximation
    with scaling and squaring for numerical stability and speed.
    """
    # Ensure input is a NumPy array of float64
    A = np.asarray(A, dtype=np.float64)
    # Basic shape check (optional but helpful for debugging)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    return expm(A)
import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to SciPy's highly optimized expm function,
    which uses a Padé approximation with scaling and squaring for numerical
    stability and speed.
    """
    # Ensure the input is a NumPy array of dtype float64
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
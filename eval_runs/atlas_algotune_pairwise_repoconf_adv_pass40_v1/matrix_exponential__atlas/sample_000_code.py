import numpy as np
from scipy.linalg import expm as _scipy_expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to SciPy's highly optimized
    `scipy.linalg.expm`, which uses a Pade approximation with
    scaling and squaring for numerical stability.
    """
    # Ensure the input is a NumPy array of float64
    A = np.asarray(A, dtype=np.float64)
    return _scipy_expm(A)
import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation uses SciPy's highly optimized expm routine,
    which performs a scaling-and-squaring algorithm with Padé
    approximants for numerical stability and speed.
    """
    # Ensure we have a numpy array of type float64
    A = np.asarray(A, dtype=np.float64)

    # Check that A is a square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix")

    return expm(A)
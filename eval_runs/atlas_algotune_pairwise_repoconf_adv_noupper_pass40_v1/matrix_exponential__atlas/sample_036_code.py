import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to scipy.linalg.expm, which uses
    a scaling-and-squaring algorithm with Pade approximants for
    numerical stability and high performance.
    """
    # Ensure the input is a square matrix of dtype float64
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if A.dtype != np.float64:
        A = A.astype(np.float64, copy=False)
    return scipy.linalg.expm(A)
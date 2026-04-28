import numpy as np
import scipy.linalg

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix ``A``.
    This implementation delegates to scipy's fast and numerically
    stable implementation based on scaling and squaring with Padé
    approximants, which matches scipy.linalg.expm(A) within tight
    tolerance.
    """
    # Ensure the input is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")
    # scipy.linalg.expm works with any real or complex dtype; cast to float64 if needed
    return scipy.linalg.expm(A.astype(np.float64))
import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    A = np.asarray(A, dtype=np.float64)
    return scipy.linalg.expm(A)
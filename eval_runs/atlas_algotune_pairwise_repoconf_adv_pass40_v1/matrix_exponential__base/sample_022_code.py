import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Use the highly optimized SciPy implementation which is both fast
    # and numerically stable. It matches scipy.linalg.expm(A) exactly.
    return scipy.linalg.expm(A)
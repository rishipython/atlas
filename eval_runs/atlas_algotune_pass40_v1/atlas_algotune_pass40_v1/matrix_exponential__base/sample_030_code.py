import numpy as np
from scipy.linalg import expm as scipy_expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # SciPy's implementation uses a highly optimized scaling and squaring
    # algorithm with Padé approximants, which is both fast and numerically stable.
    return scipy_expm(A)
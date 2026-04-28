import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # SciPy's expm uses a highly optimized scaling-and-squaring algorithm
    # with Padé approximants, which is both fast and numerically stable.
    return expm(A)
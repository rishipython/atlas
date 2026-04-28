import numpy as np
from scipy.linalg import expm as _expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # SciPy's expm implements a robust scaling-and-squaring algorithm
    # with Pade approximants, which is both fast and numerically stable.
    return _expm(A.astype(np.float64))
import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``."""
    # Use SciPy's highly optimized implementation of expm.
    # It employs scaling & squaring with a Padé approximant and is both fast and numerically stable.
    return expm(A.astype(np.float64))
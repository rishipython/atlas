import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to ``scipy.linalg.expm`` which uses a
    robust scaling-and-squaring algorithm with Padé approximants, providing
    high accuracy and speed.
    """
    return expm(A)
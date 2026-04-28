import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to :func:`scipy.linalg.expm`, which
    uses a robust scaling-and-squaring algorithm with Padé approximants
    to compute the matrix exponential efficiently and accurately.
    """
    # Ensure the input is a NumPy array of dtype float64
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
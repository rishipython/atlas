import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to ``scipy.linalg.expm`` which
    uses a highly optimized scaling-and-squaring algorithm with
    Padé approximants. The result is numerically close to the
    reference implementation and is many times faster.
    """
    # Ensure the input is a NumPy array of float64
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
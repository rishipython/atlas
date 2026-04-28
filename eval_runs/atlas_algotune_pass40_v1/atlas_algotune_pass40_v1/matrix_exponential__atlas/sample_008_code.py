import numpy as np
import scipy.linalg


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to :func:`scipy.linalg.expm`,
    which uses a scaling-and-squaring algorithm with Padé
    approximants for high accuracy and speed.
    """
    # Ensure we are working with a float64 array
    A = np.asarray(A, dtype=np.float64)
    return scipy.linalg.expm(A)
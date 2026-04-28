import numpy as np
from scipy.linalg import expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to :func:`scipy.linalg.expm`, which
    uses a highly optimized scaling‑and‑squaring algorithm with a Padé
    approximant.  The result matches `scipy.linalg.expm` to machine
    precision and is orders of magnitude faster than the naive Taylor
    series used in the reference implementation.
    """
    # Ensure the input is a NumPy array of type float64
    A = np.asarray(A, dtype=np.float64)
    return expm(A)
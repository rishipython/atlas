import numpy as np
from scipy.linalg import expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to ``scipy.linalg.expm`` which uses
    a highly optimized scaling and squaring algorithm with Padé
    approximants. It is both fast and numerically stable, matching the
    reference implementation within tight tolerances.
    """
    return expm(A)
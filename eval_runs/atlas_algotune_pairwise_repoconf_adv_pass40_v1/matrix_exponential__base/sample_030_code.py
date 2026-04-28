import numpy as np
from scipy.linalg import expm as _scipy_expm


def expm_fast(A: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a square float64 matrix ``A``.

    This implementation delegates to :func:`scipy.linalg.expm`, which
    uses a highly optimized scaling-and-squaring algorithm with Padé
    approximants and is numerically stable for a wide range of inputs.
    """
    return _scipy_expm(A)
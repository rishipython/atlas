import numpy as np
from scipy.linalg import expm as _scipy_expm

def expm_fast(A: np.ndarray) -> np.ndarray:
    """
    Return the matrix exponential of a square float64 matrix A.
    This implementation delegates to scipy.linalg.expm, which uses a
    highly optimized scaling‑and‑squaring algorithm with Padé
    approximations. The result matches scipy.linalg.expm(A) to machine
    precision for float64 inputs.
    """
    # Ensure input is float64; scipy handles other dtypes but casting
    # guarantees consistency with the reference implementation.
    A = np.asarray(A, dtype=np.float64)
    return _scipy_expm(A)
"""AlgoTune: speed up pairwise squared Euclidean distances.

Given a 2-D numpy array X of shape (N, D), compute the full (N, N) matrix
D where D[i, j] = sum_k (X[i, k] - X[j, k])**2.

The reference below is a correct-but-very-slow triple-loop Python
implementation. Your task inside the EVOLVE-BLOCK is to replace it with
a much faster implementation that returns a numerically-equal result on
the evaluator's test matrices. You are free to use numpy, scipy, numba,
or any other package that is importable.
"""

# EVOLVE-BLOCK-START
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    N, D = X.shape
    out = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(D):
                diff = X[i, k] - X[j, k]
                s += diff * diff
            out[i, j] = s
    return out
# EVOLVE-BLOCK-END

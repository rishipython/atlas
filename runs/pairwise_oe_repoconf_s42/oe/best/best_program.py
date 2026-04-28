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
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    # Use Einstein summation for a potentially faster computation of squared norms
    # Compute squared norms more directly (avoids einsum overhead)
    # Compute squared norms more directly (avoids temporary X**2)
    norms = np.einsum('ij,ij->i', X, X)
    # Compute pairwise squared distances using outer sum of norms and dot product
    # Use broadcasting for the outer sum of norms (faster than np.add.outer)
    # Compute the pairwise squared distances in-place using a single temporary array
    dist_mat = np.empty((N, N), dtype=np.float64)
    np.dot(X, X.T, out=dist_mat)          # dist_mat now holds the Gram matrix
    dist_mat *= -2                        # multiply by -2
    dist_mat += norms[:, None]            # add row norms
    dist_mat += norms[None, :]            # add column norms
    # Clip tiny negative values caused by floating‑point rounding
    np.clip(dist_mat, 0, None, out=dist_mat)
    return dist_mat
# EVOLVE-BLOCK-END

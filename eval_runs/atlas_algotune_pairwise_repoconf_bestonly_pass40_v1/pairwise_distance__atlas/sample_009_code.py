import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure float64 dtype for consistency
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Outer sum of norms gives the first two terms of the expansion
    # D[i,j] = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    out = norms[:, None] + norms[None, :] - 2.0 * X @ X.T

    # Numerical errors can lead to tiny negative values; clip them to zero
    out = np.maximum(out, 0.0)

    return out
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure we are working with float64
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)
    # Use the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist_sq = norms[:, None] + norms[None, :] - 2.0 * np.dot(X, X.T)
    # Numerical errors may produce tiny negative values; clip them to zero
    dist_sq = np.maximum(dist_sq, 0.0)
    return dist_sq
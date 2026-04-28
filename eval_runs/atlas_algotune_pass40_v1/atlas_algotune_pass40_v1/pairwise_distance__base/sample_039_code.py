import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Use the identity: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    # Outer addition of norms gives ||x_i||^2 + ||x_j||^2 for all pairs
    dist = np.add.outer(norms, norms) - 2.0 * X @ X.T

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)
    return dist
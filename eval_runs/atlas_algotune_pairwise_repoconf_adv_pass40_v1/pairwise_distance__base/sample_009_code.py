import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1, keepdims=True)  # shape (N, 1)
    # Compute squared Euclidean distance matrix via the expansion:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    prod = X @ X.T  # shape (N, N)
    out = norms + norms.T - 2.0 * prod
    # Numerical errors might produce tiny negative values; clip to zero
    out = np.maximum(out, 0.0)
    return out
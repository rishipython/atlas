import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1, keepdims=True)  # shape (N, 1)
    # Compute pairwise squared distances via broadcasting
    # D[i, j] = ||X[i]||^2 + ||X[j]||^2 - 2 * X[i]·X[j]
    dist = sq_norms + sq_norms.T - 2.0 * X @ X.T
    # Numerical errors might produce tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)
    return dist
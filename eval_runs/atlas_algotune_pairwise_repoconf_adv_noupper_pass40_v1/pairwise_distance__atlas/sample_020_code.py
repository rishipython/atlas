import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Handle empty input gracefully
    if X.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1, dtype=np.float64)  # shape (N,)

    # Compute the squared Euclidean distance matrix using the identity
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T

    # Numerical errors may lead to small negative values; clip them to zero
    np.maximum(dist2, 0.0, out=dist2)

    return dist2
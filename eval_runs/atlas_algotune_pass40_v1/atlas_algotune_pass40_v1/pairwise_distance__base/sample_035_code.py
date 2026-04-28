import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1, dtype=np.float64)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T

    # Numerical errors can make very small negatives; clip them to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
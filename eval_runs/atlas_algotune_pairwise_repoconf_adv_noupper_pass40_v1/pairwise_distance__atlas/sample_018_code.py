import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure X is a float64 array
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)
    # Use broadcasting to compute the distance matrix
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    # Numerical errors may lead to tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)
    return dist
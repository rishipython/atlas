import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)
    # Compute pairwise squared distances using the identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    dist_sq = (
        sq_norms[:, np.newaxis]
        + sq_norms[np.newaxis, :]
        - 2.0 * X.dot(X.T)
    )
    # Numerical errors may lead to tiny negative values; clip to zero.
    dist_sq = np.maximum(dist_sq, 0.0)
    return dist_sq.astype(np.float64)
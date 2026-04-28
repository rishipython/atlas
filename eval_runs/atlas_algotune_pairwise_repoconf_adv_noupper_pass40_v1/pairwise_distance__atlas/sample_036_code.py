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
    sq_norms = np.sum(X * X, axis=1)
    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    # Numerical errors can cause tiny negative values; clip them to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq
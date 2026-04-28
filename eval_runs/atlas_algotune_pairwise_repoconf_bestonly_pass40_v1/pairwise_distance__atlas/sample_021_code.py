import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    X = np.asarray(X, dtype=np.float64)
    N, _ = X.shape
    # Compute squared norms of each row
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # shape (N, 1)
    # Compute distance matrix via broadcasting
    # D[i,j] = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    dist_sq = sq_norms + sq_norms.T - 2.0 * np.dot(X, X.T)
    # Numerical errors may lead to small negative values; clip to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norm = np.sum(X * X, axis=1)
    # Use the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * X @ X.T
    # Numerical errors may produce tiny negative values; clip to zero
    return np.maximum(dist, 0.0)
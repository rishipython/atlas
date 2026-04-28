import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each point
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j
    dists = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors may produce tiny negative values; clip them to zero
    dists = np.maximum(dists, 0.0)

    return dists.astype(np.float64)
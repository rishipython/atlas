import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row vector
    norms = np.sum(X * X, axis=1)
    # Compute the Gram matrix (dot products)
    gram = X @ X.T
    # Use broadcasting to compute pairwise squared distances
    dist = norms[:, None] + norms[None, :] - 2.0 * gram
    # Numerical errors may produce tiny negative values; clip them to zero
    np.maximum(dist, 0.0, out=dist)
    return dist
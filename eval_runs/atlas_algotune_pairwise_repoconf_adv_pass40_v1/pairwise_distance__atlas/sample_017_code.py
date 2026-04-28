import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure float64 dtype for consistency
    X = np.asarray(X, dtype=np.float64, copy=False)

    # Compute squared norms of each row vector
    norms = np.sum(X * X, axis=1, keepdims=True)  # shape (N, 1)

    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist = norms + norms.T - 2.0 * gram

    # Numerical errors can produce tiny negative values; clip them to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
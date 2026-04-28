import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1, dtype=np.float64)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors can lead to tiny negative values; clip them to zero
    np.maximum(D, 0, out=D)

    return D
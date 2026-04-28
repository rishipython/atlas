import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure float64 input
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute the Gram matrix (dot product between rows)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Outer addition of norms gives ||x||^2 + ||y||^2 for all pairs
    distances = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors may produce tiny negative values; clip to zero
    distances = np.maximum(distances, 0.0)

    return distances
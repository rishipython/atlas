import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute the squared norms of each row vector
    sq_norms = np.sum(X ** 2, axis=1)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Compute the Gram matrix X·Xᵀ
    gram = np.dot(X, X.T)

    # Broadcast to compute pairwise distances
    distances = sq_norms[:, None] + sq_norms[None, :] - 2 * gram

    # Numerical errors may cause tiny negative values; clip to zero
    np.maximum(distances, 0, out=distances)

    return distances
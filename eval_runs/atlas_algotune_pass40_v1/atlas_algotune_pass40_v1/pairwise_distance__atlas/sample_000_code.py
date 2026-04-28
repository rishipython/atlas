import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is float64 for consistency
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    sq_norms = np.sum(X ** 2, axis=1)  # shape (N,)
    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)
    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    # Numerical errors may produce tiny negative values; clip them to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq
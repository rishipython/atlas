import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute the Gram matrix (dot product between rows)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    distances = sq_norms[:, None] + sq_norms[None, :] - 2 * gram

    # Numerical errors might produce tiny negative values; clip them to zero
    np.maximum(distances, 0, out=distances)

    return distances
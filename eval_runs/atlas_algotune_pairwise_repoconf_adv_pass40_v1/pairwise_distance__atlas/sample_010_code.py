import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute Gram matrix (dot product between rows)
    gram = X @ X.T  # shape (N, N)

    # Apply the formula: ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical errors can produce tiny negative values; clip to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)

    return dist_sq
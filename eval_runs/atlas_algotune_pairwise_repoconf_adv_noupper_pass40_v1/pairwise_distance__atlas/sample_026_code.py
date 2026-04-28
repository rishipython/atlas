import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses the identity
        ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    which can be computed efficiently with matrix operations.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row (shape (N,))
    sq_norms = np.sum(X * X, axis=1)
    # Compute cross term: X @ X.T (shape (N, N))
    cross = X @ X.T
    # Broadcast to compute pairwise distances
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * cross
    # Numerical errors can lead to tiny negative values; clip to zero
    dist = np.maximum(dist, 0.0)
    return dist
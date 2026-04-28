import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row (vector)
    norms = np.sum(X ** 2, axis=1)
    # Compute cross term: X @ X.T
    cross = X @ X.T
    # Use broadcasting to compute pairwise distances
    D = norms[:, None] + norms[None, :] - 2.0 * cross
    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(D, 0.0, out=D)
    return D
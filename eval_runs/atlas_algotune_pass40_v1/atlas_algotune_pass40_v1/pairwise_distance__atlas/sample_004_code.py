import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure float64 dtype for consistency
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)
    # Compute dot product matrix
    prod = X @ X.T
    # Use broadcasting to compute squared distances
    dist2 = norms[:, None] + norms[None, :] - 2.0 * prod
    # Numerical errors can produce tiny negative values; clip to zero
    np.maximum(dist2, 0.0, out=dist2)
    return dist2
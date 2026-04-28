import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses vectorised operations to compute the
    distances efficiently: D = a + a.T - 2 * X @ X.T,
    where a contains the squared norms of each row in X.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure the input is float64 for consistency
    X = np.asarray(X, dtype=np.float64)
    # Compute the dot product matrix
    dot = X @ X.T
    # Compute squared norms of each row
    sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
    # Compute squared Euclidean distances
    dist = sq_norms + sq_norms.T - 2.0 * dot
    # Numerical errors can lead to tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)
    return dist
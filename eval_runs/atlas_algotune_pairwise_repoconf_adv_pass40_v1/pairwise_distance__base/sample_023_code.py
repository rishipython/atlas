import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses vectorised matrix operations for speed.
    It returns the same result as the reference triple-loop implementation
    within floating-point tolerance.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure the input is float64 for consistency
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row (N,)
    norms = np.sum(X * X, axis=1, keepdims=True)  # shape (N, 1)

    # Compute dot product matrix (N, N)
    dot_prod = X @ X.T

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist_sq = norms + norms.T - 2.0 * dot_prod

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)

    return dist_sq
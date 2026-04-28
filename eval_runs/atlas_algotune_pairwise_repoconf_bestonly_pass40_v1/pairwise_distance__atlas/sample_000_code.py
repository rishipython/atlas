import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure we have a float64 array
    X = np.asarray(X, dtype=np.float64)
    # Sum of squares for each point (N,)
    sq_norms = np.sum(X * X, axis=1)
    # Compute pairwise squared distances using the identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    # Numerical errors may produce tiny negative values; clip them.
    D = np.maximum(D, 0.0)
    return D
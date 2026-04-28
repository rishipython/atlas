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
    sq_norms = np.sum(X * X, axis=1)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Broadcast squared norms and compute dot product
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T

    # Numerical errors may produce tiny negative values; clip to zero
    dist = np.maximum(dist, 0.0)

    return dist
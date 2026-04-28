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

    # Handle empty input
    if X.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1, keepdims=True)  # shape (N, 1)

    # Compute pairwise squared Euclidean distances using the
    # identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist = norms + norms.T - 2.0 * np.dot(X, X.T)

    # Numerical errors might produce tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
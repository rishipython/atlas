import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)  # shape (N,)
    # Compute pairwise squared distances using the identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Broadcast norms to (N,1) and (1,N) for addition
    dist2 = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
    # Numerical errors can lead to tiny negative values; clip to zero
    np.maximum(dist2, 0.0, out=dist2)
    return dist2
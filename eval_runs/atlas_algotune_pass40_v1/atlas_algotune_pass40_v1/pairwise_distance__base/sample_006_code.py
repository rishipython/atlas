import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row (shape (N,))
    sq_norms = np.sum(X * X, axis=1, dtype=np.float64)

    # Compute pairwise squared distances via the identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    # The dot product matrix has shape (N, N)
    dot_product = X @ X.T

    # Use broadcasting to add the squared norms
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot_product

    # Numerical errors might produce tiny negative values; clip them to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
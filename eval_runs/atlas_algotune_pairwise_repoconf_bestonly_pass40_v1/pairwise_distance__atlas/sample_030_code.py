import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is a 2-D array of float64
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input X must be a 2-D array")
    N, D = X.shape
    # Handle empty input
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    sq_norm = np.sum(X * X, axis=1, dtype=np.float64)  # shape (N,)

    # Compute Gram matrix (dot products between rows)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Broadcasting sq_norm to compute pairwise sums
    dists = sq_norm[:, None] + sq_norm[None, :] - 2.0 * gram

    # Numerical errors may produce tiny negative values; clip them to zero
    np.maximum(dists, 0.0, out=dists)

    return dists
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure we work with float64 for consistency
    X = np.asarray(X, dtype=np.float64)

    # Compute the squared norms of each row (shape: (N,))
    sq_norms = np.sum(X ** 2, axis=1)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Compute the full distance matrix
    dist = (
        sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    )

    # Numerical errors may produce tiny negative values; clip them to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
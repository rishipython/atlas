import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses vectorised operations to compute
    the distance matrix efficiently:
        D[i, j] = ||X[i] - X[j]||^2
                 = ||X[i]||^2 + ||X[j]||^2 - 2 * X[i] · X[j]

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure the input is float64 for consistency
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X ** 2, axis=1)
    # Compute the dot product matrix
    dot = X @ X.T
    # Broadcast to compute the squared distances
    dist = norms[:, None] + norms[None, :] - 2.0 * dot
    # Numerical errors may lead to tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)
    return dist
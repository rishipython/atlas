import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    X = np.asarray(X, dtype=np.float64)
    # Squared norms of each row
    norms = np.sum(X**2, axis=1)
    # Compute pairwise squared distances using the identity:
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    D = norms[:, None] + norms[None, :] - 2.0 * X.dot(X.T)
    # Numerical errors may produce tiny negative values; clip them to zero
    np.maximum(D, 0.0, out=D)
    return D
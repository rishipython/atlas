import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Uses the identity:
        ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is a float64 array
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute inner product matrix
    prod = X @ X.T  # shape (N, N)

    # Use broadcasting to build distance matrix
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * prod

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(D, 0.0, out=D)

    return D
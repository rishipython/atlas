import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is a float64 array
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X ** 2, axis=1)
    # Use broadcasting to construct the distance matrix
    D = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
    # Numerical errors can produce tiny negative values; clip to zero
    np.maximum(D, 0.0, out=D)
    return D
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norm = np.sum(X * X, axis=1, dtype=np.float64)  # shape (N,)

    # Compute Gram matrix (dot products)
    G = X @ X.T  # shape (N, N)

    # Apply the distance formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D = sq_norm[:, None] + sq_norm[None, :] - 2.0 * G

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(D, 0.0, out=D)

    return D
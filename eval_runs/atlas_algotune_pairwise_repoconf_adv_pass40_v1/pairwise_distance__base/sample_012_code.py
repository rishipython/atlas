import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sqnorm = np.einsum('ij,ij->i', X, X)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Compute dot product matrix once
    dot_prod = np.dot(X, X.T)

    # Broadcast to get the full distance matrix
    D = sqnorm[:, None] + sqnorm[None, :] - 2 * dot_prod

    # Numerical errors can produce tiny negative numbers; clip them to zero
    np.maximum(D, 0.0, out=D)

    return D
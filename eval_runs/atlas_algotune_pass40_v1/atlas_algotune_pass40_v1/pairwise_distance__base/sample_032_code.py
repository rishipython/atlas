import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Uses the identity:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x · y
    which allows efficient vectorised computation.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.einsum('ij,ij->i', X, X)  # shape (N,)

    # Compute the dot product matrix
    dot_prod = X @ X.T  # shape (N, N)

    # Combine terms to get squared distances
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot_prod

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
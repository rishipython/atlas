import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64 (or convertible).

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure float64 for numerical stability
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row (N,)
    norms_sq = np.sum(X * X, axis=1)

    # Compute inner product matrix (N, N)
    cross = X @ X.T

    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * cross

    # Numerical errors may produce tiny negative values; clip them to zero
    dist_sq = np.maximum(dist_sq, 0.0)

    return dist_sq
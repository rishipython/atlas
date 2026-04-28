import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (N, N) with dtype float64.
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    N, D = X.shape

    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Squared norms of each row vector
    norms = np.sum(X * X, axis=1)

    # Inner product matrix
    prod = np.dot(X, X.T)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    out = norms[:, None] + norms[None, :] - 2.0 * prod

    # Numerical errors may introduce tiny negative values; clip to zero
    np.maximum(out, 0.0, out=out)

    return out
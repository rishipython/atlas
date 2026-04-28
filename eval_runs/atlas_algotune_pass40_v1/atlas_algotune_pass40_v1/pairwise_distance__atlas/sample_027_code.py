import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64
    """
    # Ensure input is a float64 array
    X = np.asarray(X, dtype=np.float64)

    # Sum of squares for each point (shape (N,))
    sq = np.sum(X**2, axis=1)

    # Compute the squared distances using the identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist = sq[:, None] + sq[None, :] - 2.0 * X @ X.T

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
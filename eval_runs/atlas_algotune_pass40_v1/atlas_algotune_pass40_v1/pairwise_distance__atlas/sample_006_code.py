import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64.

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure we are working with float64 for consistency
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape

    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    sq_norms = np.sum(X ** 2, axis=1)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical errors can produce tiny negative values; clip them to zero
    dist = np.maximum(dist, 0.0)

    return dist
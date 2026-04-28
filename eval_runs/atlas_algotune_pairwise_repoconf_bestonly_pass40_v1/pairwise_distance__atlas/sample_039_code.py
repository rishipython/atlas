import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing squared distances.
    """
    # Ensure X is a 2-D array of float64
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # shape (N, 1)
    # Use the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist = sq_norms + sq_norms.T - 2 * X @ X.T
    # Numerical errors can make very small negative values; clip them to zero
    np.maximum(dist, 0, out=dist)
    return dist
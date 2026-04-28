import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Uses the identity:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing pairwise squared distances.
    """
    X = np.asarray(X, dtype=np.float64)
    sq_norms = np.sum(X ** 2, axis=1)  # shape (N,)
    # Compute distance matrix using broadcasting
    dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X.dot(X.T)
    # Numerical errors may produce tiny negative values; clip them to zero
    np.maximum(dist2, 0.0, out=dist2)
    return dist2
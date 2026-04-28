import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, D), dtype float64.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing squared Euclidean distances.
    """
    # Ensure input is a float64 array
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    distances = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T

    # Numerical errors can produce tiny negative values; clip them to zero
    return np.maximum(distances, 0.0)
import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D), dtype float64.

    Returns
    -------
    np.ndarray
        Output array of shape (N, N), dtype float64, where
        out[i, j] = sum_k (X[i, k] - X[j, k])**2.
    """
    # Ensure the input is a 2‑D array of float64
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape

    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist_sq = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors can produce tiny negative values; clip to zero
    dist_sq = np.maximum(dist_sq, 0.0)

    return dist_sq
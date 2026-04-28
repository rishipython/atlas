import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64

    Returns
    -------
    out : np.ndarray of shape (N, N), dtype float64
        out[i, j] = sum_k (X[i, k] - X[j, k])**2
    """
    # Ensure we are working with float64 for consistency
    X = np.asarray(X, dtype=np.float64, order='C')
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)
    # Outer sum of norms gives the diagonal part of the distance matrix
    # Then subtract twice the dot product to get squared distances
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    # Numerical errors may cause tiny negative values; clip them to zero
    np.maximum(D, 0.0, out=D)
    return D
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
    # Ensure array is contiguous and of type float64
    X = np.asarray(X, dtype=np.float64, order="C")
    N = X.shape[0]
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute Gram matrix G = X @ X.T
    G = X @ X.T

    # Squared norms of each row
    sq_norms = np.diag(G)

    # Using broadcasting to compute pairwise squared distances
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * G

    # Numerical errors might lead to tiny negative values; clip them to zero
    np.maximum(D, 0.0, out=D)

    return D
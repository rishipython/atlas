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
        Pairwise squared Euclidean distance matrix of shape (N, N).
    """
    # Ensure float64 precision
    X = np.asarray(X, dtype=np.float64)

    # Squared norms of each row: ||x_i||^2
    sq_norms = np.sum(X * X, axis=1)

    # Gram matrix: (X X^T)_{ij} = x_i · x_j
    gram = X @ X.T

    # Compute squared distances using the identity
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical errors may cause tiny negative values; clip them to zero.
    np.maximum(dist, 0.0, out=dist)

    return dist
import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        2-D array of shape (N, D) and dtype float64.

    Returns
    -------
    np.ndarray
        2-D array of shape (N, N) containing the squared Euclidean distances
        between all pairs of rows of X.
    """
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)  # shape (N,)
    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)
    # Use the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist_sq = norms[:, None] + norms[None, :] - 2.0 * gram
    # Numerical errors can lead to tiny negative values; clip them to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq
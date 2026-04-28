import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row (N,)
    sq_norms = np.einsum('ij,ij->i', X, X)  # faster than np.sum(X*X, axis=1)
    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Compute the Gram matrix
    gram = X @ X.T
    # Broadcast squared norms to form the distance matrix
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    # Numerical errors might lead to tiny negative values; clip them to zero
    np.maximum(dist_sq, 0, out=dist_sq)
    return dist_sq
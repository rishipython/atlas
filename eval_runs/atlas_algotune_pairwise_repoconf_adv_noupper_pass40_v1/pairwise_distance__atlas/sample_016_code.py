import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row vector
    sq_norm = np.sum(X * X, axis=1)
    # Compute the Gram matrix (dot products between all pairs)
    gram = X @ X.T
    # Use the identity: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    dist = sq_norm[:, None] + sq_norm[None, :] - 2 * gram
    # Numerical errors can produce tiny negative values; clip them to zero
    np.maximum(dist, 0, out=dist)
    return dist
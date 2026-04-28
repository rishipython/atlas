import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses a vectorised approach based on the identity
    ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y, which is far faster than the
    triple nested loops of the reference implementation.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure the input is a float64 array
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T

    # Use broadcasting to compute the squared distance matrix
    # D[i, j] = ||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j
    dist = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors can produce tiny negative values; clip them to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
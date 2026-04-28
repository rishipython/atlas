import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses the identity
        ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    which is computed efficiently with matrix operations.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row (vector) in X
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute the Gram matrix (dot products between all pairs)
    gram = X @ X.T  # shape (N, N)

    # Use broadcasting to compute pairwise squared distances
    # D[i, j] = norms[i] + norms[j] - 2 * gram[i, j]
    dist = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors might produce tiny negative values; clip them to zero
    np.maximum(dist, 0.0, out=dist)

    return dist
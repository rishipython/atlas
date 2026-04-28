import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure the input is a 2-D array of float64
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError("Input array must be 2-D")

    N, D = X.shape
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute the Gram matrix (inner products)
    gram = X @ X.T

    # Squared norms of each row vector
    norms = np.diagonal(gram)

    # Use broadcasting to compute pairwise squared distances
    # D[i, j] = ||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j
    dists = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors might produce tiny negative values; clip to zero
    np.maximum(dists, 0.0, out=dists)

    return dists
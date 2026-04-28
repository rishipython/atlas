import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses the vectorised formula:
        D[i, j] = ||X[i] - X[j]||^2
                 = ||X[i]||^2 + ||X[j]||^2 - 2 * X[i]·X[j]
    which is computed efficiently with NumPy operations.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure we work with a float64 array
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    sq_norms = np.sum(X ** 2, axis=1)
    # Compute the full pairwise squared distance matrix
    # Using broadcasting for the outer sum and dot product for the cross term
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * np.dot(X, X.T)
    # Numerical errors can lead to tiny negative values; clip to zero
    dist_sq = np.maximum(dist_sq, 0.0)
    return dist_sq
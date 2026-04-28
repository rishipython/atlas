import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses a vectorized formula:
        D[i, j] = ||X[i]||^2 + ||X[j]||^2 - 2 * X[i] · X[j]
    which is much faster than the reference triple-loop.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norm = np.sum(X ** 2, axis=1)  # shape (N,)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use broadcasting to form the distance matrix
    out = sq_norm[:, None] + sq_norm[None, :] - 2.0 * gram

    # Numerical errors may lead to tiny negative values; clamp to zero
    out = np.maximum(out, 0.0)
    return out
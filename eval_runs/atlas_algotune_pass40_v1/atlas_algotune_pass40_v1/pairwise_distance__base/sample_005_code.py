import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row (N,)
    sq_norms = np.sum(X * X, axis=1)
    # Compute inner product matrix (N, N)
    dot_prod = X @ X.T
    # Use broadcasting to compute pairwise squared distances
    # D[i, j] = ||X[i]||^2 + ||X[j]||^2 - 2 * <X[i], X[j]>
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot_prod
    # Numerical errors might produce tiny negative values; clip to zero
    D = np.maximum(D, 0.0)
    return D
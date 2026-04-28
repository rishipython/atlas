import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1, keepdims=True)  # shape (N, 1)
    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)
    # Use broadcasting to compute pairwise squared distances
    dists = norms + norms.T - 2.0 * gram
    # Numerical errors might produce tiny negative values; clip them to zero
    np.maximum(dists, 0.0, out=dists)
    return dists
import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is a 2-D array of floats
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use broadcasting to compute pairwise squared distances
    # D[i, j] = ||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j
    out = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors might produce tiny negative values; clip to zero
    out = np.maximum(out, 0.0)

    return out
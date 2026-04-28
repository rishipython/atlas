import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses vectorised operations to compute
    distances efficiently:
        D[i, j] = ||X[i] - X[j]||^2
                = sum(X[i]**2) + sum(X[j]**2) - 2 * X[i]·X[j]
    The result is guaranteed to be non‑negative up to numerical
    round‑off.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Sum of squares for each row: shape (N,)
    sq_norms = np.sum(X * X, axis=1)

    # Compute pairwise squared distances via broadcasting
    # D = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    # Use @ for matrix multiplication (dot product)
    cross_term = X @ X.T
    out = sq_norms[:, None] + sq_norms[None, :] - 2.0 * cross_term

    # Numerical errors can lead to tiny negative values; clip to zero
    out = np.maximum(out, 0.0)

    return out
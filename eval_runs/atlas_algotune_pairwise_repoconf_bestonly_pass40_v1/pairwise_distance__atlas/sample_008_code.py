import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is float64 for consistency
    X = np.asarray(X, dtype=np.float64)
    N, _ = X.shape

    # Handle empty input
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1)

    # Compute dot product matrix
    dot = X @ X.T

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    out = norms[:, None] + norms[None, :] - 2 * dot

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(out, 0.0, out)

    return out
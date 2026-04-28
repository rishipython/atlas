import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure float64 for numerical consistency
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row vector
    norms = np.sum(X**2, axis=1)  # shape (N,)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    D = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors may produce tiny negative values; clip them to zero
    D = np.maximum(D, 0.0)

    return D
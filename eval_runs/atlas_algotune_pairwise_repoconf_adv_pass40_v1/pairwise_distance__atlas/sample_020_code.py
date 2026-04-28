import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)  # shape (N,)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Broadcast to obtain the full pairwise matrix
    distances = (
        sq_norms[:, np.newaxis]  # shape (N, 1)
        + sq_norms[np.newaxis, :]  # shape (1, N)
        - 2.0 * np.dot(X, X.T)  # shape (N, N)
    )

    # Numerical errors can cause tiny negative values; clip them to zero
    distances = np.maximum(distances, 0.0)

    return distances
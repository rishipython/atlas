import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    norms = np.einsum('ij,ij->i', X, X)  # faster than np.sum(X**2, axis=1)
    # Compute Gram matrix
    gram = X @ X.T
    # Use broadcasting to compute pairwise squared distances
    D = norms[:, None] + norms[None, :] - 2 * gram
    # Numerical errors may lead to tiny negative values; clip them to zero
    np.maximum(D, 0, out=D)
    return D
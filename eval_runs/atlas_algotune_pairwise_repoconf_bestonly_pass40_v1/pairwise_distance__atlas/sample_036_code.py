import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Compute squared norms of each row
    norms = np.einsum('ij,ij->i', X, X)  # shape (N,)
    # Compute Gram matrix (dot product between rows)
    gram = np.dot(X, X.T)  # shape (N, N)
    # Apply the formula: ||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j
    D = norms[:, None] + norms[None, :] - 2.0 * gram
    # Numerical errors may lead to tiny negative values; clip them to zero
    np.maximum(D, 0.0, out=D)
    return D
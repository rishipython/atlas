import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure the input is a float64 array
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    norms = np.sum(X ** 2, axis=1)
    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # Compute the Gram matrix (dot product)
    gram = X @ X.T
    # Broadcast norms to form the distance matrix
    out = norms[:, None] + norms[None, :] - 2.0 * gram
    # Numerical errors may lead to tiny negative values; clip them to zero
    np.maximum(out, 0.0, out=out)
    return out
import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D), dtype float64.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing squared distances.
    """
    # Ensure input is float64 for consistency
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)          # shape (N,)

    # Compute the Gram matrix (dot products)
    gram = X @ X.T                            # shape (N, N)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical errors may produce tiny negative values; clip to zero
    dists = np.maximum(dists, 0.0)

    return dists
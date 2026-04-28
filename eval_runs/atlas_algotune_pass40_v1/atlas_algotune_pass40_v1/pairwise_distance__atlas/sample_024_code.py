import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        A square matrix of shape (N, N) where element (i, j) is the
        squared Euclidean distance between X[i] and X[j].
    """
    # Compute squared norms of each row
    norms = np.sum(X * X, axis=1, dtype=np.float64)  # shape (N,)

    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dists = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors can lead to tiny negative values; clip them to zero
    np.maximum(dists, 0.0, out=dists)

    return dists
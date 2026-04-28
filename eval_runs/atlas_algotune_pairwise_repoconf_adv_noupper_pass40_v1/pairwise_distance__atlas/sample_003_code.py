import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D), dtype float64.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) where element (i, j) is the squared Euclidean
        distance between X[i] and X[j].
    """
    # Ensure float64 to match reference
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms of each row
    sq_norms = np.sum(X * X, axis=1)  # shape (N,)
    # Compute inner products matrix
    inner = X @ X.T  # shape (N, N)
    # Use broadcasting to compute squared distances
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * inner
    # Numerical errors may produce tiny negative values; clip them to zero
    D = np.maximum(D, 0.0)
    return D
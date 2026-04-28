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
        Distance matrix of shape (N, N), dtype float64.
    """
    X = np.asarray(X, dtype=np.float64)
    # Compute squared norms for each row
    sq_norms = np.sum(X ** 2, axis=1)
    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    # Numerical errors may lead to tiny negative values; clip to zero
    np.maximum(dist2, 0.0, out=dist2)
    return dist2
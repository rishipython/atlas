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
        Distance matrix of shape (N, N) with dtype float64.
    """
    # Compute squared norms of each row
    sq_norms = np.einsum('ij,ij->i', X, X)  # shape (N,)

    # Outer sum of squared norms
    # Using broadcasting: sq_norms[:, None] + sq_norms[None, :]
    # Subtract twice the dot product matrix
    dot_prod = X @ X.T  # shape (N, N)
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * dot_prod

    # Numerical errors may produce tiny negative values; clip to zero
    np.maximum(dist, 0.0, out=dist)

    return dist.astype(np.float64)
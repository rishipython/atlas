import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64.

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64.
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    # Compute squared norms of each row
    sq_norms = np.einsum("ij,ij->i", X, X)
    # Compute pairwise squared distances via the identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * np.dot(X, X.T)
    # Numerical errors can produce tiny negative values; clip them to zero
    np.maximum(dist2, 0.0, out=dist2)
    return dist2
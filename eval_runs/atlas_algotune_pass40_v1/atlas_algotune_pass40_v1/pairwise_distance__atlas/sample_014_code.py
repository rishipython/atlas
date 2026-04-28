import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise squared Euclidean distance matrix for a set of points.

    Parameters
    ----------
    X : np.ndarray of shape (N, D) and dtype float64
        Input data matrix where each row is a point in D-dimensional space.

    Returns
    -------
    np.ndarray of shape (N, N) and dtype float64
        Distance matrix D where D[i, j] = ||X[i] - X[j]||^2.
    """
    # Ensure we are working with float64 for consistency
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row: shape (N,)
    sq_norms = np.sum(X * X, axis=1)

    # Compute the Gram matrix (dot products): shape (N, N)
    gram = X @ X.T

    # Use the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    # Broadcasting sq_norms[:, None] and sq_norms[None, :] adds the norms
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical errors can produce tiny negative values; clip them to zero
    np.maximum(D, 0.0, out=D)

    return D
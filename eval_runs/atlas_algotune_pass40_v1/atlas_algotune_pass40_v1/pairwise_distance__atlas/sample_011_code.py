import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64.

    Returns
    -------
    out : np.ndarray of shape (N, N), dtype float64.
          out[i, j] = ||X[i] - X[j]||_2**2
    """
    # Ensure input is a float64 array
    X = np.asarray(X, dtype=np.float64, order='C')
    # Compute squared norms of each row
    norms_sq = np.sum(X ** 2, axis=1)
    # Compute Gram matrix X @ X.T
    gram = X @ X.T
    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    out = norms_sq[:, None] + norms_sq[None, :] - 2.0 * gram
    # Numerical errors might produce tiny negative values; clip them to zero
    out = np.maximum(out, 0.0)
    return out
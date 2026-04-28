import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    This implementation uses the identity
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    which can be computed efficiently with matrix multiplication.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64
    """
    # Ensure we are working with a 2-D float64 array
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError("Input X must be 2-D")

    N, D = X.shape
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row vector
    norms = np.einsum("ij,ij->i", X, X)  # shape (N,)

    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use broadcasting to compute pairwise squared distances
    out = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors may cause tiny negative values; clip them to zero
    np.maximum(out, 0.0, out)

    return out
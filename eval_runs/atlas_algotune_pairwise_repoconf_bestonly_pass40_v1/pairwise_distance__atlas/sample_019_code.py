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
        Pairwise squared distance matrix of shape (N, N), dtype float64.
    """
    # Ensure input is float64 for consistency with the reference implementation
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row (N,)
    sq_norms = np.sum(X * X, axis=1)

    # Compute the full Gram matrix (inner products)
    gram = X @ X.T

    # Use the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    # Broadcast sq_norms to shape (N, 1) and (1, N) for outer addition
    dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical errors can lead to tiny negative values; clip them to zero
    np.maximum(dists, 0.0, out=dists)

    return dists
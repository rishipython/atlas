import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D) with dtype float64.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (N, N) with dtype float64.
    """
    # Ensure we work with a float64 array
    X = np.asarray(X, dtype=np.float64, order="C")

    # Sum of squares for each row: shape (N,)
    sq_norm = np.einsum("ij,ij->i", X, X)

    # Compute pairwise squared distances using the identity
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j
    # This avoids explicit loops and is highly vectorized.
    dist_sq = (
        sq_norm[:, np.newaxis]
        + sq_norm[np.newaxis, :]
        - 2.0 * X @ X.T
    )

    # Numerical errors can produce tiny negative values; clip them to zero.
    np.maximum(dist_sq, 0.0, out=dist_sq)

    return dist_sq
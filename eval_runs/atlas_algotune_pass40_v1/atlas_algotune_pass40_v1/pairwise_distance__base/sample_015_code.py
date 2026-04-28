import numpy as np


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, D) and dtype float64.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing squared Euclidean distances.
    """
    # Ensure the input is a 2-D array of float64
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError("Input array must be 2-D")

    N, D = X.shape
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row
    norms = np.einsum("ij,ij->i", X, X)  # shape (N,)

    # Compute the dot product matrix
    dot = X @ X.T  # shape (N, N)

    # Use the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    # Broadcasting norms over rows and columns
    dist_sq = norms[:, None] + norms[None, :] - 2.0 * dot

    # Numerical errors can lead to tiny negative values; clip them to zero
    np.maximum(dist_sq, 0.0, out=dist_sq)

    return dist_sq
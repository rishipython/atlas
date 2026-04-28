import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, D) with dtype float64 (or castable to it).

    Returns
    -------
    np.ndarray
        Array of shape (N, N) containing squared Euclidean distances.
    """
    # Ensure we work with float64 for consistency
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms of each row: ||x_i||^2
    sq_norm = np.sum(X * X, axis=1)

    # Compute Gram matrix G = X X^T
    G = X @ X.T

    # Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D = sq_norm[:, None] + sq_norm[None, :] - 2 * G

    # Numerical errors may lead to tiny negative values; clip to zero
    np.maximum(D, 0.0, out=D)

    return D
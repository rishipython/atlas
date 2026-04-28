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
    # Ensure the input is a float64 array
    X = np.asarray(X, dtype=np.float64)

    # Compute squared norms for each row (shape: (N,))
    sq_norm = np.sum(X * X, axis=1)

    # Compute the pairwise squared distances using the expansion
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    D = sq_norm[:, None] + sq_norm[None, :] - 2.0 * np.dot(X, X.T)

    # Numerical errors may cause tiny negative values; clip to zero
    np.maximum(D, 0.0, out=D)

    return D
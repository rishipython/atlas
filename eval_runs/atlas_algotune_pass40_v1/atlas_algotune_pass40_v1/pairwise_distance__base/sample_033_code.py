import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Args:
        X: np.ndarray of shape (N, D), dtype float64.

    Returns:
        np.ndarray of shape (N, N), dtype float64.
    """
    # Ensure input is a float64 array
    X = np.asarray(X, dtype=np.float64, copy=False)

    # Compute Gram matrix G = X @ X.T
    G = X @ X.T

    # Squared norms of each row
    sq_norm = np.diagonal(G)

    # Use broadcasting to compute pairwise squared distances
    D = sq_norm[:, None] + sq_norm[None, :] - 2.0 * G

    # Numerical errors may produce tiny negative values; clip them to zero
    D = np.maximum(D, 0.0)

    return D
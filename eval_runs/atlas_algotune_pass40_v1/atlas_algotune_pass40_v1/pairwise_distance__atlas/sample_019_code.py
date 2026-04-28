import numpy as np

def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float64.
        Input data points.

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64.
        Squared Euclidean distance matrix D where D[i, j] = ||X[i] - X[j]||^2.
    """
    # Ensure the input is a 2-D float64 array
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Input array must be 2-D, got shape {X.shape}")

    N, D = X.shape
    # Handle empty input gracefully
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute squared norms of each row (||x_i||^2)
    norms = np.sum(X * X, axis=1)  # shape (N,)

    # Compute Gram matrix (dot products)
    gram = X @ X.T  # shape (N, N)

    # Use broadcasting to compute squared distances
    # D[i, j] = norms[i] + norms[j] - 2 * gram[i, j]
    out = norms[:, None] + norms[None, :] - 2.0 * gram

    # Numerical errors can lead to tiny negative values; clip them to zero
    out = np.maximum(out, 0.0)

    return out
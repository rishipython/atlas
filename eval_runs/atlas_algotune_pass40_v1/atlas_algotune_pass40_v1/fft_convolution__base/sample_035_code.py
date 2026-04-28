import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT.

    Parameters
    ----------
    x : np.ndarray
        1-D input array of shape (N,).
    h : np.ndarray
        1-D input array of shape (M,).

    Returns
    -------
    y : np.ndarray
        Full convolution of shape (N + M - 1,).
    """
    # Ensure input is 1-D float64
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    # Handle empty inputs
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N, M = x.shape[0], h.shape[0]
    L = N + M - 1  # Length of the full convolution

    # Compute FFTs of the zero‑padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain and inverse FFT
    y = np.fft.irfft(X * H, n=L)

    return y
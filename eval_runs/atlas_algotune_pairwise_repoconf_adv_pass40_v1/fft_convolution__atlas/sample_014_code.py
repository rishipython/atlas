import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT.

    Parameters
    ----------
    x : np.ndarray
        1‑D array of length N.
    h : np.ndarray
        1‑D array of length M.

    Returns
    -------
    y : np.ndarray
        Full linear convolution of `x` and `h`, length N + M - 1.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    L = N + M - 1  # length of full convolution

    # Zero‑pad to length L and compute convolution via FFT
    # Use real FFT for efficiency
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    y = np.fft.irfft(X * H, n=L)

    return y
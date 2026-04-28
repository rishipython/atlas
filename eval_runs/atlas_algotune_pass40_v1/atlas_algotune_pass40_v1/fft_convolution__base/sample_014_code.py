import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1
    # Use real FFT for speed and numerical stability
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    Y = X * H
    y = np.fft.irfft(Y, n=L)
    return y
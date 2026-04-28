import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size

    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    out_len = N + M - 1
    # Next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # Compute real FFTs, multiply, inverse FFT
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    Y = X * H
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact convolution length
    return y[:out_len]
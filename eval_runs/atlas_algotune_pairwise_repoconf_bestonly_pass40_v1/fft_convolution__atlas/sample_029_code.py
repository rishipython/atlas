import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1-D and of float64 dtype
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.shape[0], h.shape[0]
    n = N + M - 1

    # Handle empty inputs
    if n == 0:
        return np.array([], dtype=np.float64)

    # Choose an efficient FFT length
    nfft = np.fft.next_fast_len(n)

    # Compute FFTs and multiply
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the exact convolution length
    return y[:n]
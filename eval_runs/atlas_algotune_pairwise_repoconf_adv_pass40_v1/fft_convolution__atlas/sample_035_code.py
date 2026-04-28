import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return np.convolve(x, h, mode='full') but MUCH faster using FFT."""
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    n_out = N + M - 1

    if n_out == 0:
        return np.empty((0,), dtype=np.float64)

    # Choose an efficient FFT length
    L = np.fft.next_fast_len(n_out)

    # Compute FFTs and multiply
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    Y = X * H

    # Inverse FFT and truncate to the desired output length
    y = np.fft.irfft(Y, n=L)
    return y[:n_out]
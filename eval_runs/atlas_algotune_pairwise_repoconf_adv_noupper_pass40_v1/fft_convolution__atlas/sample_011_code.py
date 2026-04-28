import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Handle empty inputs like numpy's convolve
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N = x.size
    M = h.size
    L = N + M - 1  # length of full convolution

    # Compute FFTs using real FFT for efficiency
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.irfft(Y, n=L)

    return y
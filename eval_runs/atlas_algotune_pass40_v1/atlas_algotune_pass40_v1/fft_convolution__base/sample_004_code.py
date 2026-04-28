import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    N = x.shape[0]
    M = h.shape[0]
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    L = N + M - 1

    # Compute the real FFTs of the zero‑padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Elementwise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain the convolution result
    y = np.fft.irfft(Y, n=L)

    return y
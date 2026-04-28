import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return np.convolve(x, h, mode='full') but MUCH faster."""
    N = x.shape[0]
    M = h.shape[0]
    # Handle empty inputs
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Next power of two for efficient FFT
    L = 1 << ((N + M - 1 - 1).bit_length())

    # Forward FFTs (real-to-complex)
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Elementwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the required length
    return y[: N + M - 1]
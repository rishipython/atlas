import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1

    # Handle empty input gracefully
    if L == 0:
        return np.array([], dtype=np.float64)

    # Choose an FFT size that is a power of two for speed
    nfft = 1 << (L - 1).bit_length()

    # Compute FFTs of both signals padded to nfft
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Elementwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the expected full convolution length
    return y[:L]
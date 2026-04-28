import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays
    using an FFT-based approach for speed.
    Equivalent to np.convolve(x, h, mode='full').
    """
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N, M = x.shape[0], h.shape[0]

    # Handle empty input cases
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    out_len = N + M - 1

    # Choose FFT length as next power of two for efficiency
    L = 1 << (out_len - 1).bit_length()

    # Perform real FFTs, multiply, and inverse transform
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    Y = X * H
    y = np.fft.irfft(Y, n=L)

    # Truncate to the required output length
    return y[:out_len]
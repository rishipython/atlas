import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but faster for large inputs.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    # Length of the full convolution
    out_len = N + M - 1

    # Choose FFT length: next power of two for speed (optional)
    fft_len = 1 << (out_len - 1).bit_length()

    # Perform real FFTs, multiply, then inverse real FFT
    X = np.fft.rfft(x, fft_len)
    H = np.fft.rfft(h, fft_len)
    Y = X * H
    y = np.fft.irfft(Y, fft_len)

    # Truncate to the exact output length
    return y[:out_len]
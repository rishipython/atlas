import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but faster for large inputs.
    """
    # Ensure input dtype is float64
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    size = N + M - 1

    # Handle trivial cases
    if size == 0:
        return np.array([], dtype=np.float64)

    # Choose FFT size as next power of two for speed (optional)
    nfft = 1 << (size - 1).bit_length()

    # Compute FFTs and multiply
    X = np.fft.rfft(x, n=nfft)
    H = np.fft.rfft(h, n=nfft)
    Y = X * H

    # Inverse FFT and truncate to the required length
    y = np.fft.irfft(Y, n=nfft)
    return y[:size].astype(np.float64)
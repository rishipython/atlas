import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but considerably faster for
    large inputs.  The result has dtype float64.
    """
    # Ensure inputs are 1‑D and float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    size = N + M - 1

    # Handle empty inputs
    if size == 0:
        return np.empty(0, dtype=np.float64)

    # Choose FFT length: next power of two >= size for speed
    L = 1 << (size - 1).bit_length()

    # Compute FFTs of zero‑padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.irfft(Y, n=L)

    # Trim to the required output length
    return y[:size]
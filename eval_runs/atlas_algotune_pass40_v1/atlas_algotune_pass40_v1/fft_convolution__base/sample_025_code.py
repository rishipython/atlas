import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays using FFT.

    Parameters
    ----------
    x : np.ndarray
        Input array of length N.
    h : np.ndarray
        Input array of length M.

    Returns
    -------
    y : np.ndarray
        Full convolution of length N + M - 1.
    """
    # Handle empty inputs as numpy.convolve does
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N, M = x.shape[0], h.shape[0]
    size = N + M - 1

    # Pad to next power of two for efficient FFT
    nfft = 1 << (size - 1).bit_length()

    # FFT of zero‑padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the desired length
    y = np.fft.irfft(Y, nfft)[:size]
    return y
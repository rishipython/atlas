import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT.

    Parameters
    ----------
    x : np.ndarray
        1‑D array of length N.
    h : np.ndarray
        1‑D array of length M.

    Returns
    -------
    y : np.ndarray
        Full convolution of length N + M - 1.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    if out_len == 0:
        return np.array([], dtype=np.float64)

    # Choose a transform length that is a power of two for speed
    L = 1 << ((out_len - 1).bit_length())

    # Compute real FFTs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT and truncate to the desired length
    y = np.fft.irfft(Y, n=L)[:out_len]

    return y
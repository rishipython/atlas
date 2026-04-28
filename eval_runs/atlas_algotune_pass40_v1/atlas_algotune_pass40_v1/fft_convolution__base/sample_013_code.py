import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based algorithm for speed.

    Parameters
    ----------
    x : np.ndarray
        Input array of length N.
    h : np.ndarray
        Input array of length M.

    Returns
    -------
    y : np.ndarray
        Convolution result of length N + M - 1.
    """
    N, M = x.shape[0], h.shape[0]
    L = N + M - 1
    if L == 0:
        return np.array([], dtype=np.float64)

    # Choose a power‑of‑two length >= L for efficient FFT.
    L2 = 1 << (L - 1).bit_length()

    # Forward FFTs (real‑to‑complex) with zero‑padding to length L2.
    X = np.fft.rfft(x, n=L2)
    H = np.fft.rfft(h, n=L2)

    # Pointwise multiplication in frequency domain.
    Y = X * H

    # Inverse FFT (complex‑to‑real) and truncate to desired length.
    y = np.fft.irfft(Y, n=L2)[:L]
    return y
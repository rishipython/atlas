import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based algorithm.

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
    # Ensure input is 1‑D float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    n_out = N + M - 1

    # Next power of two for efficient FFT (optional but usually faster)
    L = 1 << (n_out - 1).bit_length()

    # Compute forward FFTs (real‑to‑complex)
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution (real‑to‑real)
    y_full = np.fft.irfft(Y, n=L)

    # Truncate to the exact convolution length
    return y_full[:n_out]
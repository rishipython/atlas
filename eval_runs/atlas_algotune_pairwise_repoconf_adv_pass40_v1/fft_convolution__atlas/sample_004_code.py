import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based algorithm, which is much faster than a
    naive double‑loop implementation.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N,).
    h : np.ndarray
        Input array of shape (M,).

    Returns
    -------
    y : np.ndarray
        Convolution result of shape (N + M - 1,).
    """
    N, M = x.shape[0], h.shape[0]
    L = N + M - 1

    # Choose FFT length as the next power of two for speed
    L_fft = 1 << (L - 1).bit_length()

    # FFT of zero‑padded inputs
    X = np.fft.fft(x, n=L_fft)
    H = np.fft.fft(h, n=L_fft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and take the real part
    y_full = np.fft.ifft(Y).real

    # Truncate to the exact convolution length
    return y_full[:L]
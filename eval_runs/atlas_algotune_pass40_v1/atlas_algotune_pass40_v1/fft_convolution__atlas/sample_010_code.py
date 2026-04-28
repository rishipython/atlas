import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D real float64 arrays
    using an FFT‑based algorithm.  The result is numerically close to
    np.convolve(x, h, mode='full').

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N,).
    h : np.ndarray
        Input array of shape (M,).

    Returns
    -------
    y : np.ndarray
        Convolution of x and h, shape (N + M - 1,).
    """
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    size = N + M - 1

    # Choose FFT length as next power of two for efficiency.
    L = 1 << (size - 1).bit_length()

    # Compute real FFTs of zero‑padded inputs.
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain.
    Y = X * H

    # Inverse real FFT to obtain convolution result.
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact linear convolution length.
    return y[:size]
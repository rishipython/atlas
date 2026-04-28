import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays `x` and `h`
    using an FFT-based approach for speed.

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
    # Ensure inputs are 1-D and float64
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1

    # Handle empty inputs
    if L == 0:
        return np.empty(0, dtype=np.float64)

    # Use a power-of-two FFT length for speed
    nfft = 1 << (L - 1).bit_length()

    # Compute forward FFTs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution
    y = np.fft.irfft(Y, nfft)[:L]

    return y
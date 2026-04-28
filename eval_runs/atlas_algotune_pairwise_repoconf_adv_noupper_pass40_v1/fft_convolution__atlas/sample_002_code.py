import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT.

    Parameters
    ----------
    x : np.ndarray
        1‑D input array of dtype float64.
    h : np.ndarray
        1‑D kernel array of dtype float64.

    Returns
    -------
    y : np.ndarray
        1‑D array of length ``len(x) + len(h) - 1`` containing the
        convolution result.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')

    N = x.shape[0]
    M = h.shape[0]
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    out_len = N + M - 1

    # Use the next power of two for efficient FFT
    nfft = 1 << (out_len - 1).bit_length()

    # Compute FFTs of zero‑padded signals
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Point‑wise multiplication in frequency domain
    Y = np.fft.irfft(X * H, nfft)

    # Truncate to the required output length
    return Y[:out_len]
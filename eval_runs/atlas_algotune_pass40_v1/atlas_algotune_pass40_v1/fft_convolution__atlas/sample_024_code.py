import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Compute the full linear convolution of two 1‑D arrays using FFT.

    Parameters
    ----------
    x : np.ndarray
        First input array, of length N.
    h : np.ndarray
        Second input array, of length M.

    Returns
    -------
    y : np.ndarray
        Array of length N + M - 1 containing the convolution.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.shape[0], h.shape[0]
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the FFT (next power of two >= N + M - 1)
    conv_len = N + M - 1
    L = 1 << (conv_len - 1).bit_length()

    # Compute FFTs and multiply in frequency domain
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact length
    return y[:conv_len]
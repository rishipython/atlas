import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT.

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
    # Ensure inputs are 1‑D and of float64 dtype
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    size = N + M - 1

    # Next power of two for efficient FFT (optional but speeds up for large arrays)
    fft_size = 1 << (size - 1).bit_length()

    # Compute FFTs
    X = np.fft.rfft(x, fft_size)
    H = np.fft.rfft(h, fft_size)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.irfft(Y, fft_size)[:size]

    return y
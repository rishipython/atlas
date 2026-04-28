import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D real float64 arrays.
    This implementation uses FFT-based convolution for speed.

    Parameters
    ----------
    x : np.ndarray
        Input array of length N.
    h : np.ndarray
        Input array of length M.

    Returns
    -------
    y : np.ndarray
        Array of length N + M - 1 containing the convolution.
    """
    # Ensure input arrays are 1-D float64
    x = np.asarray(x, dtype=np.float64, order='C').ravel()
    h = np.asarray(h, dtype=np.float64, order='C').ravel()

    N = x.size
    M = h.size
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the linear convolution result
    conv_len = N + M - 1

    # Next power of two for efficient FFT (optional, but improves speed)
    L = 1 << (conv_len - 1).bit_length()

    # Compute FFTs using real FFT (rfft) which is faster for real inputs
    X = np.fft.rfft(x, L)
    H = np.fft.rfft(h, L)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain the convolution result
    y_full = np.fft.irfft(Y, L)

    # Truncate to the exact convolution length
    return y_full[:conv_len].copy()
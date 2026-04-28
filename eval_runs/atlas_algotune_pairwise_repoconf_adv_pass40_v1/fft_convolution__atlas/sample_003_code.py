import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1-D float64 arrays using FFT.

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
    # Ensure input arrays are 1-D and float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    L = N + M - 1

    if L == 0:
        return np.empty(0, dtype=np.float64)

    # Choose FFT length as next power of two for speed
    fft_len = 1 << (L - 1).bit_length()

    # Compute forward FFTs
    X = np.fft.rfft(x, n=fft_len)
    H = np.fft.rfft(h, n=fft_len)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to desired length
    y = np.fft.irfft(Y, n=fft_len)[:L]

    return y
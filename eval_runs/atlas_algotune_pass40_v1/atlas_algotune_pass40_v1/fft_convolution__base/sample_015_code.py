import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays
    using an FFT-based method for speed.

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
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.shape[0]
    M = h.shape[0]
    n = N + M - 1

    # Handle the trivial case where one of the inputs is empty
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Determine the next power-of-two size for efficient FFT
    size = 1 << (n - 1).bit_length()

    # Compute the FFTs (real-to-complex) of the zero-padded inputs
    X = np.fft.rfft(x, size)
    H = np.fft.rfft(h, size)

    # Pointwise multiplication in the frequency domain
    Y = X * H

    # Inverse FFT to obtain the convolution result (real output)
    y = np.fft.irfft(Y, size)[:n]

    return y
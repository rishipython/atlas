import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Compute the full linear convolution of two 1-D real-valued signals
    using an FFT-based method.  This is equivalent to
    np.convolve(x, h, mode='full') but is typically much faster for
    large inputs.

    Parameters
    ----------
    x : np.ndarray
        1-D input array of length N.
    h : np.ndarray
        1-D input array of length M.

    Returns
    -------
    y : np.ndarray
        Full linear convolution of `x` and `h`, of length N+M-1.
    """
    # Ensure inputs are 1-D arrays of float64
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("Both inputs must be 1-D arrays.")
    N, M = x.size, h.size
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    # Length of the output sequence
    L = N + M - 1

    # Next power of two for efficient FFT (optional)
    size = 1 << (L - 1).bit_length()

    # Compute the FFTs of the zero-padded signals
    X = np.fft.rfft(x, n=size)
    H = np.fft.rfft(h, n=size)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the convolution result
    y = np.fft.irfft(Y, n=size)

    # Truncate to the correct length
    return y[:L]
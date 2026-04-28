import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but faster for large inputs.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N, M = x.shape[0], h.shape[0]
    # Length of the full convolution
    conv_len = N + M - 1
    # Compute next power of two for efficient FFT (optional, but often faster)
    L = 1 << (conv_len - 1).bit_length()
    # Perform real FFTs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    # Element-wise multiplication in frequency domain
    Y = X * H
    # Inverse real FFT to get convolution result
    y = np.fft.irfft(Y, n=L)
    # Truncate to the exact convolution length
    return y[:conv_len]
import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    This implementation is numerically equivalent to np.convolve(x, h, mode='full')
    but is much faster for large inputs.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    L = N + M - 1  # desired output length

    if L == 0:
        return np.array([], dtype=np.float64)

    # Next power of two for efficient FFT (optional, but helps speed)
    nfft = 1 << (L - 1).bit_length()

    # Compute FFTs of zero-padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Elementwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the exact output length
    return y[:L]
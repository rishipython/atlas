import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Ensure 1-D arrays
    x = x.ravel()
    h = h.ravel()

    N = x.size
    M = h.size
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    n = N + M - 1

    # Next power of two for efficient FFT
    nfft = 1 << (n - 1).bit_length()

    # FFT of the zero-padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the time-domain convolution
    y = np.fft.irfft(Y, nfft)[:n]

    return y
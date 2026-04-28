import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Handle empty inputs: numpy.convolve returns an empty array when either input is empty
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    n = N + M - 1

    # Choose FFT size: next power of two for speed (optional but usually faster)
    size = 1 << (n - 1).bit_length()

    # Compute real FFTs of the zero-padded inputs
    X = np.fft.rfft(x, size, dtype=np.float64)
    H = np.fft.rfft(h, size, dtype=np.float64)

    # Elementwise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get time-domain convolution
    y = np.fft.irfft(Y, size, dtype=np.float64)

    # Truncate to the exact convolution length
    return y[:n]
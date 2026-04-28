import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N = x.shape[0]
    M = h.shape[0]
    size = N + M - 1

    # Use the next fast length for FFT
    nfft = np.fft.next_fast_len(size)

    # Compute FFTs using real FFT for speed
    X = np.fft.rfft(x, n=nfft)
    H = np.fft.rfft(h, n=nfft)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain convolution result
    y = np.fft.irfft(Y, n=nfft)

    # Truncate to the exact convolution length
    return y[:size]
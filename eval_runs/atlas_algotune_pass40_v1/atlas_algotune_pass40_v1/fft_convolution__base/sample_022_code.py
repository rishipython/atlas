import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    N = x.shape[0]
    M = h.shape[0]
    size = N + M - 1
    # Next power of two for efficient FFT
    nfft = 1 << (size - 1).bit_length()
    # Compute FFTs using real FFT for speed
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    # Elementwise multiplication in frequency domain
    Y = X * H
    # Inverse real FFT and truncate to the required length
    y = np.fft.irfft(Y, nfft)
    return y[:size]
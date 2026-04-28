import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1

    # Use the next power of two for efficient FFT
    L2 = 1 << (L - 1).bit_length()

    # Compute FFTs, multiply, and inverse FFT
    X = np.fft.fft(x, L2)
    H = np.fft.fft(h, L2)
    y = np.fft.ifft(X * H).real[:L]

    return y
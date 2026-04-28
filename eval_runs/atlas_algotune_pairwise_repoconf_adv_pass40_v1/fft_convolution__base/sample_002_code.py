import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N, M = x.shape[0], h.shape[0]

    # If either input is empty, the full convolution is empty
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    L = N + M - 1

    # Use real FFT for speed (inputs are real)
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Elementwise multiplication in frequency domain and inverse transform
    y = np.fft.irfft(X * H, n=L)

    return y
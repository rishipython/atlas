import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.size
    M = h.size

    # Handle empty inputs
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of full convolution
    L = N + M - 1

    # FFT‑based convolution
    X = np.fft.fft(x, n=L)
    H = np.fft.fft(h, n=L)
    y = np.fft.ifft(X * H).real

    return y
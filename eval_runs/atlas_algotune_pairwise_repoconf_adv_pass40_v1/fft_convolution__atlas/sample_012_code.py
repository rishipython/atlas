import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT for speed."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the linear convolution result
    L = N + M - 1

    # Use real FFT for efficiency
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element-wise multiplication in frequency domain
    Y = np.fft.irfft(X * H, n=L)

    return Y.astype(np.float64)
import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    N, M = x.size, h.size
    if N == 0 or M == 0:
        # Empty convolution
        return np.empty(N + M - 1, dtype=np.float64)

    size = N + M - 1

    # Use real FFTs for speed (inputs are real)
    X = np.fft.rfft(x, n=size)
    H = np.fft.rfft(h, n=size)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain the convolution result
    y = np.fft.irfft(Y, n=size)

    return y
import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return np.convolve(x, h, mode='full') but MUCH faster using FFT."""
    # Ensure input arrays are 1‑D float64
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N, M = x.shape[0], h.shape[0]
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    out_len = N + M - 1

    # Use real FFTs for speed
    X = np.fft.rfft(x, n=out_len)
    H = np.fft.rfft(h, n=out_len)

    # Multiply in frequency domain and inverse transform
    y = np.fft.irfft(X * H, n=out_len)

    return y
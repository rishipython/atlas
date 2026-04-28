import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure 1‑D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    # Handle empty input (np.convolve returns empty array)
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    L = N + M - 1  # length of the full convolution

    # Use real FFT for efficiency (inputs are real)
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get the convolution result
    y = np.fft.irfft(Y, n=L)

    return y
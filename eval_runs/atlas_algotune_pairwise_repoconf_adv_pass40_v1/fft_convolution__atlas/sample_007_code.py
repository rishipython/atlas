import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D arrays using FFT."""
    # Ensure input is 1‑D and dtype float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.shape[0], h.shape[0]
    L = N + M - 1  # Length of the full convolution

    # Handle empty inputs
    if L <= 0:
        return np.array([], dtype=np.float64)

    # Choose FFT length as next power of two for speed
    L_fft = 1 << (L - 1).bit_length()

    # Compute FFTs of zero‑padded inputs
    X = np.fft.rfft(x, n=L_fft)
    H = np.fft.rfft(h, n=L_fft)

    # Point‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.irfft(Y, n=L_fft)

    # Return only the first L samples (full convolution)
    return y[:L]
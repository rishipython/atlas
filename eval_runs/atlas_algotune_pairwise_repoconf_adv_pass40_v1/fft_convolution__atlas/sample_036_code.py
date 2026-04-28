import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT."""
    # Ensure input is float64 and 1‑D
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.shape[0], h.shape[0]
    # Length of the result
    out_len = N + M - 1

    # Next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # FFT of zero‑padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact output length
    return y[:out_len]
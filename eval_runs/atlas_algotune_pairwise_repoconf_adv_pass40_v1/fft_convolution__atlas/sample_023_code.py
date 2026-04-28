import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT."""
    N, M = x.shape[0], h.shape[0]

    # Handle empty inputs
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the output
    out_len = N + M - 1

    # Find the next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # Compute real FFTs of the zero‑padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain the convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact output length
    return y[:out_len].astype(np.float64)
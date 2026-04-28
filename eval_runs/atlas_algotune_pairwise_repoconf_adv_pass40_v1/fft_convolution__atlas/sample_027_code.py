import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays.
    This implementation uses an FFT-based approach for speed.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    # Length of the full convolution result
    L = N + M - 1

    # Handle empty inputs
    if L <= 0:
        return np.array([], dtype=np.float64)

    # Use the next power of two for efficient FFT (optional)
    nfft = 1 << (L - 1).bit_length()

    # Compute FFTs of the zero‑padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the correct length
    y = np.fft.irfft(Y, nfft)[:L]

    return y
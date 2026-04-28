import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]

    # Handle empty input gracefully
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    # Length of the full convolution
    conv_len = N + M - 1

    # Choose a fast FFT length (next fast length >= conv_len)
    L = np.fft.next_fast_len(conv_len)

    # Compute FFTs of zero‑padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the time‑domain convolution
    y = np.fft.irfft(Y, n=L)

    # Trim to the exact convolution length
    return y[:conv_len]
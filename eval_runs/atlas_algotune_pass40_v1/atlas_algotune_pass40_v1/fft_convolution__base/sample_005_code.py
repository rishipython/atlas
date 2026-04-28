import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    out_len = N + M - 1

    # Choose FFT length as next power of two >= out_len for speed
    n_fft = 1 << (out_len - 1).bit_length()

    # Compute FFTs (real‑to‑complex for speed)
    X = np.fft.rfft(x, n_fft)
    H = np.fft.rfft(h, n_fft)

    # Pointwise multiplication and inverse FFT
    y = np.fft.irfft(X * H, n_fft)

    # Truncate to the exact convolution length
    return y[:out_len].copy()
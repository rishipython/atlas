import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64, copy=False)
    h = np.asarray(h, dtype=np.float64, copy=False)

    N = x.size
    M = h.size
    out_len = N + M - 1

    # If either input is empty, return an empty array
    if out_len <= 0:
        return np.empty(0, dtype=np.float64)

    # Use the next power of two for efficient FFT
    fft_len = 1 << (out_len - 1).bit_length()

    # Compute FFTs of zero‑padded signals
    X = np.fft.rfft(x, fft_len)
    H = np.fft.rfft(h, fft_len)

    # Point‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the required length
    y = np.fft.irfft(Y, fft_len, dtype=np.float64)[:out_len]

    return y
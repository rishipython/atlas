import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    # Length of the full convolution
    out_len = N + M - 1

    if out_len == 0:
        return np.array([], dtype=np.float64)

    # Next power of two for efficient FFT
    nfft = 1 << (out_len - 1).bit_length()

    # Compute real FFTs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT and truncate to the required length
    y = np.fft.irfft(Y, nfft)[:out_len]

    return y
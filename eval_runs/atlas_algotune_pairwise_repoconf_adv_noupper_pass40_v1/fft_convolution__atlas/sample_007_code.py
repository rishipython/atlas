import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1  # desired output length

    if N == 0 or M == 0:
        # One of the inputs is empty – return an array of zeros
        return np.zeros(L, dtype=np.float64)

    # Next power of two for efficient FFT (optional but speeds up large arrays)
    nfft = 1 << (L - 1).bit_length()

    # Compute FFTs of both signals, zero‑padded to nfft
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain the convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the exact output length (in case nfft > L)
    return y[:L]
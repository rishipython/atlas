import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays.

    This implementation uses the FFT for speed.  It is numerically
    equivalent to ``np.convolve(x, h, mode='full')`` for real inputs.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    out_len = N + M - 1

    # Next power of two for efficient FFT (optional but speeds up large arrays)
    n_fft = 1 << (out_len - 1).bit_length()

    # Compute FFTs of zero‑padded signals
    X = np.fft.rfft(x, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the required output length
    y = np.fft.irfft(Y, n=n_fft)[:out_len]

    return y
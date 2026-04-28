import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT.

    This implementation is numerically equivalent to
    ``np.convolve(x, h, mode='full')`` but is substantially faster for
    large input sizes.

    Parameters
    ----------
    x : np.ndarray
        1‑D array of floats.
    h : np.ndarray
        1‑D array of floats.

    Returns
    -------
    y : np.ndarray
        1‑D array of floats, length ``len(x) + len(h) - 1``.
    """
    # Ensure inputs are 1‑D and float64
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N, M = x.size, h.size
    L = N + M - 1

    # Handle empty inputs
    if L <= 0:
        return np.empty(0, dtype=np.float64)

    # Next power of two for efficient FFT
    L_fft = 1 << (L - 1).bit_length()

    # FFT of zero‑padded signals
    X = np.fft.fft(x, L_fft)
    H = np.fft.fft(h, L_fft)

    # Point‑wise multiplication and inverse FFT
    y_full = np.fft.ifft(X * H).real

    # Truncate to the linear convolution length
    return y_full[:L]
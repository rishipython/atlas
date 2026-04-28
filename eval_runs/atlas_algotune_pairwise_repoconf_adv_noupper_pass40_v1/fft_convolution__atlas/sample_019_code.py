import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays.

    Uses an FFT‑based method for speed while preserving numerical
    accuracy. The result is equivalent to ``np.convolve(x, h, mode='full')``.
    """
    # Ensure input arrays are float64 views
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N, M = x.shape[0], h.shape[0]
    L = N + M - 1

    # Choose FFT length as the next power of two >= L for speed
    nfft = 1 << (L - 1).bit_length()

    # Forward real FFTs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT and truncate to the desired output length
    y = np.fft.irfft(Y, nfft)
    return y[:L]
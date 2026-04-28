import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure input is 1-D float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()
    N = x.size
    M = h.size
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Desired output length
    L = N + M - 1

    # Choose FFT size: next power of two greater than or equal to L
    nfft = 1 << (L - 1).bit_length()

    # Compute FFTs of the zero‑padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the correct length
    y = np.fft.irfft(Y, nfft, dtype=np.float64)[:L]

    return y
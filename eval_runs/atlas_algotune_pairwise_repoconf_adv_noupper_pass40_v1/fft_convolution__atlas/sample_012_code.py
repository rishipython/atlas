import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    N = x.shape[0]
    M = h.shape[0]

    # Handle empty inputs
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    L = N + M - 1  # length of the full convolution

    # Helper to find next power of two for efficient FFT
    def _next_pow2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    size = _next_pow2(L)

    # Compute FFTs of the zero‑padded signals
    X = np.fft.rfft(x, n=size)
    H = np.fft.rfft(h, n=size)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y_full = np.fft.irfft(Y, n=size)

    # Truncate to the exact convolution length
    return y_full[:L]
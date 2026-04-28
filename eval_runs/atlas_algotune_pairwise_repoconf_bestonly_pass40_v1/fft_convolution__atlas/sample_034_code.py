import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of 1-D float64 arrays x and h.
    This implementation uses an FFT-based approach for speed.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    # Handle empty inputs
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution result
    L = N + M - 1

    # Next power of two for efficient FFT (optional, but speeds up large convolutions)
    size = 1 << (L - 1).bit_length()

    # FFT of zero‑padded inputs
    X = np.fft.rfft(x, n=size)
    H = np.fft.rfft(h, n=size)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result, trimmed to length L
    y = np.fft.irfft(Y, n=size, dtype=np.float64)[:L]

    return y
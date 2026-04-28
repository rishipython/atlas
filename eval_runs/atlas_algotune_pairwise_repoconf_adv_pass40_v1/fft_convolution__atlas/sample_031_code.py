import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    # Length of the FFT: next power of two >= N + M - 1
    conv_len = N + M - 1
    L = 1 << (conv_len - 1).bit_length()

    # FFT of zero-padded inputs using real FFT for efficiency
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact convolution length
    return y[:conv_len]
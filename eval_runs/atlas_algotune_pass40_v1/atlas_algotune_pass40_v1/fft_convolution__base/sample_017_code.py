import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT for speed."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    N, M = x.size, h.size
    # Handle empty inputs: np.convolve([] , a) returns empty array
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    # Length of the full convolution result
    L = N + M - 1

    # Find next power of two for efficient FFT
    fft_len = 1 << (L - 1).bit_length()

    # Compute FFTs
    X = np.fft.rfft(x, n=fft_len)
    H = np.fft.rfft(h, n=fft_len)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and take real part
    y = np.fft.irfft(Y, n=fft_len)

    # Truncate to the required length
    return y[:L].astype(np.float64)
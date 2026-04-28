import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order="C").ravel()
    h = np.asarray(h, dtype=np.float64, order="C").ravel()

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Determine FFT length: next power of two >= out_len
    L = 1 << (out_len - 1).bit_length()

    # Compute FFTs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element-wise multiplication and inverse FFT
    y_full = np.fft.irfft(X * H, n=L)

    # Truncate to the exact output length
    return y_full[:out_len]
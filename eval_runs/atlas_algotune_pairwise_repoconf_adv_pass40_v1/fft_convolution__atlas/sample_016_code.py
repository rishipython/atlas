import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, copy=False)
    h = np.asarray(h, dtype=np.float64, copy=False)

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Handle empty inputs: numpy returns an empty array
    if out_len == 0:
        return np.empty(0, dtype=np.float64)

    # Use FFT-based convolution
    # Choose a fast FFT length (next power of two or a more efficient size)
    L = np.fft.next_fast_len(out_len)

    # Compute FFTs of zero-padded inputs
    X = np.fft.rfft(x, L)
    H = np.fft.rfft(h, L)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the convolution result
    y = np.fft.irfft(Y, L)

    # Truncate to the expected output length
    return y[:out_len].astype(np.float64)
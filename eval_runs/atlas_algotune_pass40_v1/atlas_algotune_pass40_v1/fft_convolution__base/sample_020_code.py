import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C').ravel()
    h = np.asarray(h, dtype=np.float64, order='C').ravel()

    N = x.shape[0]
    M = h.shape[0]
    size = N + M - 1

    # Choose a fast FFT size (next power of 2 or a fast length)
    nfft = np.fft.next_fast_len(size)

    # Compute FFTs of the padded signals
    X = np.fft.rfft(x, n=nfft)
    H = np.fft.rfft(h, n=nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain the convolution result
    y = np.fft.irfft(Y, n=nfft)

    # Trim to the exact output length
    return y[:size]
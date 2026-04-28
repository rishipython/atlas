import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the linear convolution result
    conv_len = N + M - 1

    # Next power of two for efficient FFT (optional but speeds up)
    fft_len = 1 << (conv_len - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, n=fft_len)
    H = np.fft.fft(h, n=fft_len)

    # Element-wise multiplication and inverse FFT
    Y = np.fft.ifft(X * H).real

    # Truncate to the exact convolution length
    return Y[:conv_len]
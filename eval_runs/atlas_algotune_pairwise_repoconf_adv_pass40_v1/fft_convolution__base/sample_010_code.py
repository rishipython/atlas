import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1
    # Use a fast FFT length (power of two or other efficient size)
    L_fft = np.fft.next_fast_len(L)
    # Compute FFTs of zero-padded inputs
    X = np.fft.rfft(x, n=L_fft)
    H = np.fft.rfft(h, n=L_fft)
    # Element-wise multiplication in frequency domain
    Y = X * H
    # Inverse FFT to get linear convolution
    y = np.fft.irfft(Y, n=L_fft)
    # Truncate to the exact convolution length
    return y[:L]
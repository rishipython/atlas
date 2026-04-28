import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.size
    M = h.size
    L = N + M - 1

    # Choose an FFT size that is fast for the underlying implementation
    nfft = np.fft.next_fast_len(L)

    # Compute FFTs of zero-padded signals
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the desired output length
    y = np.fft.irfft(Y, nfft)[:L]
    return y
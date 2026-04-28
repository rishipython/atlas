import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure input arrays are 1-D float64
    x = np.asarray(x, dtype=np.float64, order='C').ravel()
    h = np.asarray(h, dtype=np.float64, order='C').ravel()

    N = x.shape[0]
    M = h.shape[0]
    # Desired output length
    out_len = N + M - 1

    # Next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # Compute forward FFTs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the correct length
    return y[:out_len]